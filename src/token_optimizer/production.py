"""Production-ready token optimizer pipeline.

This module is the one-step integration surface for applications: shadow telemetry,
rollout gating, provider probing, smart compression, safe fallback, HTTP request,
and optimization metadata are handled behind a single OpenAI-compatible call.

v2 Enhancement: Integrated CCR (Compression with Content Recall) from Headroom.
When compression is applied, original content is stored in a CompressionStore
so the downstream LLM can retrieve it on-demand via hash markers.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from token_optimizer.config import OptimizerConfig
from token_optimizer.core.compression_store import CompressionStore
from token_optimizer.core.prompt_reorderer import compute_prefix_hash, reorder_messages, strip_dynamic_fields
from token_optimizer.core.smart_compressor import SmartCompressor, estimate_tokens_from_messages
from token_optimizer.models.model_config import estimate_cache_savings, get_model_profile

RolloutMode = Literal["off", "shadow", "auto", "on"]


@dataclass
class RolloutGate:
    """Deterministic production gate for enabling real smart compression.

    The gate is intentionally conservative. In ``auto`` mode, smart compression is
    enabled only when shadow telemetry predicts positive savings and no fallback.
    This makes the product usable immediately while retaining safe rollout control.
    """

    mode: RolloutMode = "auto"
    min_estimated_savings_pct: float = 20.0
    allow_protected: bool = True
    max_protected_span_count: int = 64
    require_probe_available: bool = False

    def decide(self, telemetry: dict[str, Any]) -> dict[str, Any]:
        if self.mode == "off":
            return {"enabled": False, "reason": "rollout_mode_off"}
        if self.mode == "shadow":
            return {"enabled": False, "reason": "rollout_mode_shadow"}
        if self.mode == "on":
            return {"enabled": True, "reason": "rollout_mode_on"}

        if not telemetry.get("would_call_smart"):
            return {"enabled": False, "reason": telemetry.get("would_fallback_reason") or "shadow_not_profitable"}
        savings = telemetry.get("estimated_savings_pct")
        if savings is None or float(savings) < self.min_estimated_savings_pct:
            return {"enabled": False, "reason": "estimated_savings_below_threshold", "estimated_savings_pct": savings}
        if not self.allow_protected and telemetry.get("policy_mode") == "protected":
            return {"enabled": False, "reason": "protected_policy_disabled"}
        if int(telemetry.get("protected_span_count") or 0) > self.max_protected_span_count:
            return {"enabled": False, "reason": "too_many_protected_spans"}
        route = telemetry.get("route") or {}
        probe = route.get("probe") or {}
        if self.require_probe_available and probe.get("enabled") and not probe.get("available"):
            return {"enabled": False, "reason": "probe_not_available"}
        return {"enabled": True, "reason": "auto_gate_passed", "estimated_savings_pct": savings}


@dataclass
class ProductionOptimizerConfig:
    """Configuration for the production optimizer surface."""

    model: str = "mimo-v2.5-pro"
    api_key: str = ""
    base_url: str = ""
    rollout: RolloutGate = field(default_factory=RolloutGate)
    enable_model_probe: bool = True
    enable_prefix_reorder: bool = True
    enable_prefix_cache: bool = True
    enable_ccr: bool = True
    ccr_max_entries: int = 50
    ccr_default_ttl: float = 300.0
    request_timeout: float = 120.0
    smart_min_rule_tokens: int = 128
    attach_metadata: bool = True

    @classmethod
    def from_optimizer_config(
        cls,
        config: OptimizerConfig,
        *,
        rollout: RolloutGate | None = None,
        enable_model_probe: bool = True,
    ) -> "ProductionOptimizerConfig":
        return cls(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            rollout=rollout or RolloutGate(),
            enable_model_probe=enable_model_probe,
            enable_prefix_reorder=config.enable_prefix_reorder,
            enable_prefix_cache=config.enable_prefix_cache,
            request_timeout=config.request_timeout,
        )

    def __post_init__(self) -> None:
        if not self.base_url:
            self.base_url = OptimizerConfig(model=self.model, api_key=self.api_key).base_url


class ProductionOptimizer:
    """OpenAI-compatible production token optimization wrapper.

    The public entrypoint is ``chat_completions_create``. It never lets compression
    failures break the main request: any optimizer error falls back to the safest
    available message path and records why in ``_optimization``.
    """

    def __init__(self, config: ProductionOptimizerConfig | None = None, **kwargs: Any) -> None:
        self.config = config or ProductionOptimizerConfig(**kwargs)
        self.profile = get_model_profile(self.config.model)
        self.smart = SmartCompressor(
            main_model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=min(self.config.request_timeout, 60.0),
            min_rule_tokens_for_smart=self.config.smart_min_rule_tokens,
            enable_model_probe=self.config.enable_model_probe,
        )
        self._last_prefix_hash: str | None = None
        # CCR: Compression with Content Recall
        self.ccr_store = CompressionStore(
            max_entries=self.config.ccr_max_entries,
            default_ttl=self.config.ccr_default_ttl,
        ) if self.config.enable_ccr else None

    def chat_completions_create(self, **kwargs: Any) -> dict[str, Any]:
        started = time.perf_counter()
        original_messages = kwargs.get("messages", []) or []
        model = kwargs.get("model", self.config.model)
        tools = kwargs.get("tools")
        optimization: dict[str, Any] = {
            "production_optimizer": True,
            "model": model,
            "rollout_mode": self.config.rollout.mode,
            "errors": [],
        }

        try:
            cleaned = strip_dynamic_fields(original_messages)
        except Exception as e:
            cleaned = original_messages
            optimization["errors"].append(f"strip_dynamic_fields_failed: {str(e)[:160]}")

        system_text = ""
        for message in cleaned:
            if message.get("role") == "system":
                system_text = message.get("content", "")
                break

        try:
            shadow = self.smart.shadow_evaluate(cleaned, system_text=system_text)
        except Exception as e:
            shadow = {"mode": "shadow", "would_call_smart": False, "would_fallback_reason": "shadow_exception", "reason": str(e)[:160]}
            optimization["errors"].append(f"shadow_failed: {str(e)[:160]}")

        gate = self.config.rollout.decide(shadow)
        optimization["shadow"] = shadow
        optimization["rollout_gate"] = gate

        l1_meta: dict[str, Any] = {"mode": "not_run"}
        ccr_meta: dict[str, Any] = {"enabled": self.config.enable_ccr}

        # CCR: Store original content before compression
        ccr_hash_map: dict[str, str] = {}  # content_prefix → hash_key
        if self.ccr_store and gate.get("enabled", False):
            for msg in cleaned:
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 50:
                    hash_key, _ = self.ccr_store.store(
                        original_text=content,
                        compressed_text=content,  # updated after annotate
                    )
                    # Use first 80 chars as lookup key for content matching
                    ccr_hash_map[content[:80]] = hash_key
            ccr_meta["stored_count"] = len(ccr_hash_map)

        if gate["enabled"]:
            try:
                optimized_messages, l1_meta = self.smart.compress(cleaned, system_text=system_text)
                # CCR: Annotate compressed messages with retrieval markers
                if self.ccr_store and ccr_hash_map:
                    optimized_messages = self._ccr_annotate(
                        optimized_messages, ccr_hash_map
                    )
            except Exception as e:
                optimized_messages = cleaned
                l1_meta = {"mode": "safe_passthrough_repair", "reason": f"smart_compress_exception: {str(e)[:160]}"}
                optimization["errors"].append(l1_meta["reason"])
        else:
            # Use the same deterministic rule compressor when available; if it fails,
            # keep original cleaned messages.
            try:
                optimized_messages, rule_meta = self.smart.rule_compressor.compress_messages(cleaned, system_text=system_text)
                l1_meta = {"mode": "rule_only", "reason": gate["reason"], "rule_compression": rule_meta}
            except Exception as e:
                optimized_messages = cleaned
                l1_meta = {"mode": "safe_passthrough_repair", "reason": f"rule_compress_exception: {str(e)[:160]}"}
                optimization["errors"].append(l1_meta["reason"])

        if self.config.enable_prefix_reorder:
            try:
                final_messages, reorder_meta = reorder_messages(optimized_messages, tools)
            except Exception as e:
                final_messages = optimized_messages
                reorder_meta = {"prefix_hash": "", "prefix_tokens_est": 0, "error": str(e)[:160]}
                optimization["errors"].append(f"reorder_failed: {str(e)[:160]}")
        else:
            final_messages = optimized_messages
            reorder_meta = {"prefix_hash": "", "prefix_tokens_est": 0}

        prefix_hash = compute_prefix_hash(final_messages)
        cache_stable = self._last_prefix_hash is not None and prefix_hash == self._last_prefix_hash
        self._last_prefix_hash = prefix_hash
        prefix_tokens = int(reorder_meta.get("prefix_tokens_est") or 0)
        cache_savings = estimate_cache_savings(self.profile, prefix_tokens)

        payload = {
            **{k: v for k, v in kwargs.items() if k not in ("messages", "tools")},
            "model": model,
            "messages": final_messages,
        }
        if tools is not None:
            payload["tools"] = tools

        response = self._sync_request(payload)
        usage = response.get("usage") or {}
        final_prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        if final_prompt_tokens is None:
            final_prompt_tokens = estimate_tokens_from_messages(final_messages, model=model)

        original_tokens = estimate_tokens_from_messages(original_messages, model=model)
        optimized_tokens = estimate_tokens_from_messages(final_messages, model=model)
        token_saved_pct = ((original_tokens - optimized_tokens) / original_tokens * 100) if original_tokens else 0.0

        optimization.update({
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "input_tokens_original_est": original_tokens,
            "input_tokens_optimized_est": optimized_tokens,
            "input_tokens_api": final_prompt_tokens,
            "token_saved_pct_est": round(token_saved_pct, 2),
            "l1_smart_compression": l1_meta,
            "prefix_reorder": reorder_meta,
            "prefix_hash": prefix_hash,
            "cache_stable": cache_stable,
            "cache_savings_estimate": cache_savings,
            "final_path": "smart" if l1_meta.get("mode") == "smart" else "rule_or_passthrough",
            "ccr": ccr_meta,
        })
        if self.config.attach_metadata:
            response["_optimization"] = optimization
        return response

    def _sync_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.request_timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        ) as client:
            response = client.post("/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()

    def _ccr_annotate(
        self,
        compressed_messages: list[dict[str, Any]],
        ccr_hash_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Annotate compressed messages with CCR retrieval markers.

        Uses content-prefix matching instead of index-based matching to
        correctly handle cases where SmartCompressor changes message count/order.

        Args:
            compressed_messages: Messages after compression.
            ccr_hash_map: Mapping from content_prefix (first 80 chars of original)
                         to CCR hash key.

        Returns:
            Annotated compressed messages with retrieval markers.
        """
        if not self.ccr_store or not ccr_hash_map:
            return compressed_messages

        result = []
        for msg in compressed_messages:
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 10:
                result.append(msg)
                continue

            # Try to match by checking if compressed content is a prefix of
            # any stored original, or if original starts with compressed content
            matched_hash = None
            for orig_prefix, hash_key in ccr_hash_map.items():
                if content[:80] == orig_prefix:
                    # Exact prefix match (compression preserved start)
                    matched_hash = hash_key
                    break
                if orig_prefix.startswith(content[:60]):
                    # Original starts with compressed content
                    matched_hash = hash_key
                    break
                if content[:60] in orig_prefix:
                    # Compressed content is substring of original prefix
                    matched_hash = hash_key
                    break

            if matched_hash and self.ccr_store.has(matched_hash):
                marker = f" [TO:retrieve hash={matched_hash}]"
                if marker not in content:
                    result.append({**msg, "content": content + marker})
                else:
                    result.append(msg)
            else:
                result.append(msg)
        return result

    def ccr_retrieve(self, hash_key: str) -> str | None:
        """Retrieve original content by CCR hash key.

        This is the public API for LLMs (or tool-call handlers) to retrieve
        the original content when they see a [TO:retrieve hash=xxx] marker.

        Args:
            hash_key: The 12-char hex hash from the retrieval marker.

        Returns:
            Original text if found, None otherwise.
        """
        if not self.ccr_store:
            return None
        return self.ccr_store.retrieve(hash_key)
