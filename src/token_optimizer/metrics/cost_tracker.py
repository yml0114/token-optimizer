"""Cost tracker — transparent cost reporting for every API call.

Records actual token usage per request and calculates cost savings
from cache hits vs fresh input.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from token_optimizer.models.model_config import ModelProfile, get_model_profile


@dataclass
class CostEntry:
    """One request's cost record."""
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    raw_input_cost: float
    actual_input_cost: float
    output_cost: float
    total_cost: float
    savings: float
    cache_hit: bool


class CostTracker:
    """Tracks costs across requests and reports savings."""

    def __init__(self, model: str):
        self._profile = get_model_profile(model)
        self._entries: list[CostEntry] = []

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> CostEntry:
        """Record one API call's token usage and compute costs."""
        now = time.time()

        # Cost without cache = all tokens at input price
        raw_input_cost = (input_tokens / 1_000_000) * self._profile.input_price_per_m

        # Actual cost = (input - cached) at full price + cached at cache price
        non_cached_input = input_tokens - cached_tokens
        actual_input_cost = (
            (non_cached_input / 1_000_000) * self._profile.input_price_per_m
            + (cached_tokens / 1_000_000) * self._profile.cache_price_per_m
        )

        # Output cost (no optimization on output in v1.0)
        output_cost = (output_tokens / 1_000_000) * self._profile.output_price_per_m

        total = actual_input_cost + output_cost
        raw_total = raw_input_cost + output_cost
        savings = raw_total - total
        cache_hit = cached_tokens > 0

        entry = CostEntry(
            timestamp=now,
            model=self._profile.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            raw_input_cost=round(raw_input_cost, 8),
            actual_input_cost=round(actual_input_cost, 8),
            output_cost=round(output_cost, 8),
            total_cost=round(total, 8),
            savings=round(savings, 8),
            cache_hit=cache_hit,
        )
        self._entries.append(entry)
        return entry

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    def summary(self) -> dict:
        """Aggregate cost summary across all tracked requests."""
        if not self._entries:
            return {
                "model": self._profile.name,
                "total_requests": 0,
                "message": "No requests tracked yet",
            }

        total_raw = sum(e.raw_input_cost + e.output_cost for e in self._entries)
        total_actual = sum(e.total_cost for e in self._entries)
        total_savings = total_raw - total_actual
        hits = sum(1 for e in self._entries if e.cache_hit)

        return {
            "model": self._profile.name,
            "total_requests": len(self._entries),
            "cache_hits": hits,
            "cache_hit_rate": round(hits / len(self._entries) * 100, 1),
            "total_input_tokens": sum(e.input_tokens for e in self._entries),
            "total_output_tokens": sum(e.output_tokens for e in self._entries),
            "total_cached_tokens": sum(e.cached_tokens for e in self._entries),
            "raw_total_cost": round(total_raw, 6),
            "actual_total_cost": round(total_actual, 6),
            "total_savings": round(total_savings, 6),
            "savings_pct": round(total_savings / total_raw * 100, 1) if total_raw > 0 else 0,
            "avg_cost_per_request": round(
                total_actual / len(self._entries), 8
            ),
            "avg_raw_cost_per_request": round(
                total_raw / len(self._entries), 8
            ),
        }

    def format_summary(self) -> str:
        """Human-readable summary string."""
        s = self.summary()
        if s.get("total_requests", 0) == 0:
            return "No requests tracked yet."

        lines = [
            f"═══ Token Optimizer Cost Report ═══",
            f"Model:       {s['model']}",
            f"Requests:    {s['total_requests']} ({s['cache_hits']} cache hits, {s['cache_hit_rate']}% hit rate)",
            f"Input:       {s['total_input_tokens']:,} tokens ({s['total_cached_tokens']:,} cached)",
            f"Output:      {s['total_output_tokens']:,} tokens",
            f"──────────────────────────────────",
            f"Without opt: ${s['raw_total_cost']:.6f}",
            f"With opt:    ${s['actual_total_cost']:.6f}",
            f"Saved:       ${s['total_savings']:.6f} ({s['savings_pct']}%)",
            f"Avg/req:     ${s['avg_cost_per_request']:.8f} (was ${s['avg_raw_cost_per_request']:.8f})",
            f"══════════════════════════════════",
        ]
        return "\n".join(lines)
