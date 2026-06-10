"""Main TokenOptimizer client — OpenAI-compatible interface.

Usage:
    optimizer = TokenOptimizer(model="deepseek-v4-flash", api_key="sk-...")
    response = optimizer.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}],
    )
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from token_optimizer.config import OptimizerConfig
from token_optimizer.core.prompt_reorderer import (
    compute_prefix_hash,
    reorder_messages,
    strip_dynamic_fields,
)
from token_optimizer.core.signal_noise import InputCompressor, CompressionLevel
from token_optimizer.core.cache_manager import CacheManager
from token_optimizer.metrics.cost_tracker import CostTracker
from token_optimizer.models.model_config import (
    ModelProfile,
    get_model_profile,
    estimate_cache_savings,
)


class ChatCompletions:
    """Interface mimicking OpenAI's chat.completions."""

    def __init__(self, client: "TokenOptimizer"):
        self._client = client

    def create(self, **kwargs) -> dict[str, Any]:
        return self._client._chat_completions_create(**kwargs)


class TokenOptimizer:
    """Universal LLM Token Optimization Engine.

    Wraps an API client with L0 (prefix reorder) + L2 (prefix cache) optimizations.
    API-compatible with OpenAI's SDK interface.
    """

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        api_key: str = "",
        base_url: str = "",
        config: OptimizerConfig | None = None,
    ):
        if config is None:
            config = OptimizerConfig(model=model, api_key=api_key, base_url=base_url)

        self._config = config
        self._profile: ModelProfile = get_model_profile(config.model)

        # Core components
        self._cache = CacheManager()
        self._cost_tracker = CostTracker(config.model)

        # L1: Input compressor
        comp_level = CompressionLevel(config.compression_level)
        self._compressor = InputCompressor(level=comp_level)

        # HTTP client
        self._http = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.request_timeout,
            headers=self._build_headers(),
        )

        # Public interface
        self.chat = ChatCompletions(self)

        # State
        self._last_prefix_hash: str | None = None

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

    def _chat_completions_create(self, **kwargs) -> dict[str, Any]:
        """Internal: optimized chat completion call.

        Applies L0 (reorder) + L2 (cache tracking) optimizations.
        """
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools", None)
        model = kwargs.get("model", self._config.model)

        # ── Step 1: Strip dynamic fields that break prefix ──
        cleaned = strip_dynamic_fields(messages)

        # ── Step 1.5: L1 — Compress input by removing noise ──
        compression_meta = {"compressed": False, "savings_pct": 0}
        if self._config.enable_input_compression and cleaned:
            system_text = ""
            for m in cleaned:
                if m.get("role") == "system":
                    system_text = m.get("content", "")
                    break
            cleaned, compression_meta = self._compressor.compress_messages(
                cleaned, system_text=system_text
            )

        # ── Step 2: L0 — Reorder for cache alignment ──
        if self._config.enable_prefix_reorder:
            reordered, reorder_meta = reorder_messages(cleaned, tools)
        else:
            reordered = cleaned
            reorder_meta = {"prefix_hash": "", "prefix_tokens_est": 0}

        # ── Step 3: Check cache eligibility ──
        prefix_hash = compute_prefix_hash(reordered)
        is_cache_stable = (
            self._last_prefix_hash is not None
            and prefix_hash == self._last_prefix_hash
        )
        self._last_prefix_hash = prefix_hash

        # ── Step 4: Build API payload ──
        payload = {
            "model": model,
            "messages": reordered,
            **{k: v for k, v in kwargs.items() if k not in ("messages", "tools")},
        }

        # ── Step 5: Estimate cache savings ──
        prefix_tokens = reorder_meta.get("prefix_tokens_est", 0)
        cache_savings = estimate_cache_savings(self._profile, prefix_tokens)

        # ── Step 6: Make API call ──
        # TODO: In v1.0, we use sync httpx. Async will be added.
        # For now, use the sync client internally via httpx.Client
        raw_response = self._sync_request(payload)

        # ── Step 7: Parse response tokens ──
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0

        usage = raw_response.get("usage", {})
        if usage:
            input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

            # Some APIs report cached tokens directly
            cached_tokens = usage.get("cached_tokens", 0) or usage.get(
                "prompt_cache_hit_tokens", 0
            )

        # Estimate: if API doesn't report cache hits, use our own heuristic
        if cached_tokens == 0 and is_cache_stable:
            # Assume stable prefix → likely cached
            cached_tokens = min(input_tokens, prefix_tokens)

        cache_hit = cached_tokens > 0

        # ── Step 8: Track everything ──
        cost_entry = self._cost_tracker.record(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        )

        self._cache.record_request(
            prefix_hash=prefix_hash,
            cache_hit=cache_hit,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            model=model,
            estimated_cost=cost_entry.total_cost,
            prefix_tokens_est=prefix_tokens,
        )

        # ── Step 9: Attach optimization metadata to response ──
        raw_response["_optimization"] = {
            "prefix_hash": prefix_hash,
            "cache_stable": is_cache_stable,
            "cache_hit": cache_hit,
            "cached_tokens": cached_tokens,
            "input_compression": reorder_meta,
            "l1_compression": compression_meta,
            "cost": {
                "raw": cache_savings["raw_cost"],
                "actual": cost_entry.actual_input_cost,
                "savings": cost_entry.savings,
                "savings_pct": cache_savings["savings_pct"],
            },
        }

        return raw_response

    def _sync_request(self, payload: dict) -> dict[str, Any]:
        """Make synchronous HTTP request (v1.0 sync mode)."""
        import httpx as httpx_sync

        with httpx_sync.Client(
            base_url=self._config.base_url,
            timeout=self._config.request_timeout,
            headers=self._build_headers(),
        ) as client:
            resp = client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()

    @property
    def metrics(self) -> CostTracker:
        return self._cost_tracker

    @property
    def cache_manager(self) -> CacheManager:
        return self._cache

    def summary(self) -> str:
        """Human-readable optimization summary."""
        return self._cost_tracker.format_summary()

    def close(self):
        """Clean up resources."""
        import asyncio
        try:
            asyncio.get_running_loop()
            # In async context
        except RuntimeError:
            pass
