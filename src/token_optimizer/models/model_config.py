"""Model-specific pricing and API configuration.

Each model has unique cache pricing rules. This module centralizes that knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    """Immutable model cost & cache profile."""

    name: str
    input_price_per_m: float    # $ per million input tokens
    output_price_per_m: float   # $ per million output tokens
    cache_price_per_m: float    # $ per million cached tokens
    cache_write_price_per_m: float  # $ per million tokens written to cache
    max_context: int            # max context window
    cache_min_prefix: int       # min tokens to trigger cache
    cache_granularity: int      # cache storage unit in tokens
    supports_chat_prefix: bool  # supports Chat Prefix Completion


# ──────────────────────────────────────────────
#  Known Model Profiles (2026-06-11 pricing)
# ──────────────────────────────────────────────

DEEPSEEK_V4_FLASH = ModelProfile(
    name="deepseek-v4-flash",
    input_price_per_m=0.14,
    output_price_per_m=0.28,
    cache_price_per_m=0.0028,
    cache_write_price_per_m=0.0,      # auto cache, no write fee
    max_context=1_000_000,
    cache_min_prefix=1024,
    cache_granularity=64,
    supports_chat_prefix=True,
)

DEEPSEEK_V4_PRO = ModelProfile(
    name="deepseek-v4-pro",
    input_price_per_m=0.435,
    output_price_per_m=0.87,
    cache_price_per_m=0.003625,
    cache_write_price_per_m=0.0,
    max_context=1_000_000,
    cache_min_prefix=1024,
    cache_granularity=64,
    supports_chat_prefix=True,
)

MIMO_V2_FLASH = ModelProfile(
    name="mimo-v2-flash",
    input_price_per_m=0.10,
    output_price_per_m=0.30,
    cache_price_per_m=0.01,
    cache_write_price_per_m=0.0,      # free writes (limited time)
    max_context=256_000,
    cache_min_prefix=1024,
    cache_granularity=64,
    supports_chat_prefix=False,
)

MIMO_V2_5 = ModelProfile(
    name="mimo-v2.5",
    input_price_per_m=0.14,
    output_price_per_m=0.28,
    cache_price_per_m=0.0028,
    cache_write_price_per_m=0.0,      # free writes (limited time, Token Plan)
    max_context=1_000_000,
    cache_min_prefix=1024,
    cache_granularity=64,
    supports_chat_prefix=False,
)

MIMO_V2_5_PRO = ModelProfile(
    name="mimo-v2.5-pro",
    input_price_per_m=1.00,
    output_price_per_m=3.00,
    cache_price_per_m=0.20,
    cache_write_price_per_m=0.0,      # free writes (limited time)
    max_context=1_000_000,
    cache_min_prefix=1024,
    cache_granularity=64,
    supports_chat_prefix=False,
)


# ──────────────────────────────────────────────
#  Lookup Table
# ──────────────────────────────────────────────

_PROFILES: dict[str, ModelProfile] = {
    p.name: p for p in [
        DEEPSEEK_V4_FLASH,
        DEEPSEEK_V4_PRO,
        MIMO_V2_FLASH,
        MIMO_V2_5,
        MIMO_V2_5_PRO,
    ]
}

# Alias map for flexible matching
_ALIASES: dict[str, str] = {
    "deepseek-v4": "deepseek-v4-flash",
    "deepseek-flash": "deepseek-v4-flash",
    "deepseek-pro": "deepseek-v4-pro",
    "mimo-flash": "mimo-v2-flash",
    "mimo-v2-flash": "mimo-v2-flash",
    "mimo-pro": "mimo-v2.5-pro",
    "mimo-v2.5-pro": "mimo-v2.5-pro",
    "mimo-v2.5": "mimo-v2.5",
    "mimo": "mimo-v2.5",
}


def get_model_profile(model: str) -> ModelProfile:
    """Get the cost profile for a model.

    Raises:
        ValueError: if model is not recognized.
    """
    key = model.lower().strip().replace(" ", "-")
    # Direct match
    if key in _PROFILES:
        return _PROFILES[key]
    # Alias match
    if key in _ALIASES:
        return _PROFILES[_ALIASES[key]]
    # Partial match: find first profile where model name is substring
    for name, profile in _PROFILES.items():
        if name in key or key in name:
            return profile
    raise ValueError(
        f"Unknown model '{model}'. "
        f"Known models: {', '.join(sorted(_PROFILES.keys()))}"
    )


def estimate_cache_savings(profile: ModelProfile, prefix_tokens: int) -> dict:
    """Estimate cost savings from cache hit vs fresh input.

    Returns dict with raw_cost, cached_cost, savings, savings_pct.
    """
    raw_cost = (prefix_tokens / 1_000_000) * profile.input_price_per_m
    cached_cost = (prefix_tokens / 1_000_000) * profile.cache_price_per_m
    savings = raw_cost - cached_cost
    savings_pct = (savings / raw_cost * 100) if raw_cost > 0 else 0

    return {
        "raw_cost": round(raw_cost, 8),
        "cached_cost": round(cached_cost, 8),
        "savings": round(savings, 8),
        "savings_pct": round(savings_pct, 1),
        "write_cost": round(
            (prefix_tokens / 1_000_000) * profile.cache_write_price_per_m, 8
        ),
    }
