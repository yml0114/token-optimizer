"""Provider model probing for same-platform cheap compressor discovery.

The probe is intentionally safe by default:
- no network call unless caller explicitly invokes ``probe()`` with base_url/api_key;
- provider failures become structured diagnostics, not exceptions;
- discovered models are ranked by profitability hints and context capacity;
- pricing is a conservative local catalog that can be overridden by callers later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx

from token_optimizer.core.smart_compressor import CheapModelOption, ModelRoute, find_route


@dataclass(frozen=True)
class ProbeModelInfo:
    """Normalized model record returned by a provider /models endpoint."""

    id: str
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProbeResult:
    """Structured provider probe result."""

    provider: str
    main_model: str
    available: bool
    source: str
    models_seen: tuple[str, ...]
    cheap_options: tuple[CheapModelOption, ...]
    route: ModelRoute | None
    failure_reason: str | None = None


@dataclass(frozen=True)
class PriceHint:
    input_price: float
    output_price: float
    max_context: int = 1_000_000
    cross_generation: bool = False
    note: str = ""


# Conservative local hints. Unknown cheap-looking models can still be surfaced with
# guarded fallback prices; profit guard in SmartCompressor remains the final gate.
PRICE_HINTS: dict[str, PriceHint] = {
    "mimo-v2-flash": PriceHint(0.10, 0.30, max_context=256_000, cross_generation=True, note="MiMo cross-generation cheap compressor"),
    "mimo-v2.5": PriceHint(0.14, 0.28, max_context=1_000_000, note="MiMo standard model"),
    "mimo-v2.5-pro": PriceHint(1.00, 3.00, max_context=1_000_000, note="MiMo Pro main model"),
    "deepseek-v4-flash": PriceHint(0.14, 0.28, max_context=1_000_000),
    "deepseek-v4-pro": PriceHint(0.435, 0.87, max_context=1_000_000),
    "qwen-turbo": PriceHint(0.05, 0.20, max_context=1_000_000),
    "qwen-plus": PriceHint(0.40, 1.20, max_context=1_000_000),
    "qwen-max": PriceHint(2.40, 9.60, max_context=1_000_000),
    "gpt-4o-mini": PriceHint(0.15, 0.60, max_context=128_000),
    "gpt-4o": PriceHint(2.50, 10.00, max_context=128_000),
    "claude-3-haiku": PriceHint(0.25, 1.25, max_context=200_000),
    "claude-3-opus": PriceHint(15.00, 75.00, max_context=200_000),
    "claude-3.5-sonnet": PriceHint(3.00, 15.00, max_context=200_000),
}

CHEAP_MODEL_PATTERNS = (
    "flash",
    "mini",
    "turbo",
    "lite",
    "haiku",
    "small",
    "compress",
)

PROVIDER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("mimo", "mimo"),
    ("xiaomimimo", "mimo"),
    ("deepseek", "deepseek"),
    ("qwen", "qwen"),
    ("dashscope", "qwen"),
    ("openai", "openai"),
    ("anthropic", "anthropic"),
)


def infer_provider(base_url: str = "", model: str = "") -> str:
    """Infer provider family from base_url or model name."""
    text = f"{base_url} {model}".lower()
    for pattern, provider in PROVIDER_PATTERNS:
        if pattern in text:
            return provider
    model_lower = model.lower()
    if model_lower.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if model_lower.startswith("claude-"):
        return "anthropic"
    return "unknown"


def normalize_models_payload(payload: Any) -> tuple[ProbeModelInfo, ...]:
    """Normalize OpenAI-like /models payloads and plain lists."""
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        return ()
    models: list[ProbeModelInfo] = []
    for item in data:
        if isinstance(item, str):
            model_id = item
            raw = {"id": item}
        elif isinstance(item, dict):
            model_id = str(item.get("id") or item.get("name") or item.get("model") or "")
            raw = item
        else:
            continue
        model_id = model_id.strip()
        if model_id:
            models.append(ProbeModelInfo(id=model_id, raw=raw))
    return tuple(models)


def _provider_compatible(provider: str, main_model: str, candidate: str) -> bool:
    if provider == "unknown":
        return True
    return provider in candidate.lower() or provider in main_model.lower() or provider in {"openai", "anthropic"}


def _looks_cheap(model_id: str) -> bool:
    lower = model_id.lower()
    return any(pattern in lower for pattern in CHEAP_MODEL_PATTERNS)


def _price_hint(model_id: str) -> PriceHint:
    key = model_id.lower()
    if key in PRICE_HINTS:
        return PRICE_HINTS[key]
    # Conservative fallback for unknown cheap-looking candidates. Profit guard will
    # reject it if the main model is too cheap or the ratio is not enough.
    if _looks_cheap(key):
        return PriceHint(0.20, 0.60, max_context=128_000, note="fallback cheap-model price hint; verify provider pricing")
    return PriceHint(1.00, 3.00, max_context=128_000, note="fallback main-model price hint; verify provider pricing")


def _same_generation_family_score(main_model: str, candidate: str) -> int:
    main_tokens = set(re.findall(r"[a-z]+|\d+(?:\.\d+)?", main_model.lower()))
    cand_tokens = set(re.findall(r"[a-z]+|\d+(?:\.\d+)?", candidate.lower()))
    return len(main_tokens & cand_tokens)


def discover_cheap_options(
    main_model: str,
    models: tuple[ProbeModelInfo, ...] | list[ProbeModelInfo],
    *,
    provider: str = "unknown",
    limit: int = 5,
) -> tuple[CheapModelOption, ...]:
    """Rank cheap-looking same-provider model candidates."""
    main_lower = main_model.lower()
    candidates: list[tuple[float, CheapModelOption]] = []
    for model in models:
        model_id = model.id.strip()
        lower = model_id.lower()
        if not model_id or lower == main_lower:
            continue
        if not _looks_cheap(lower):
            continue
        if not _provider_compatible(provider, main_lower, lower):
            continue
        hint = _price_hint(lower)
        score = 0.0
        score += _same_generation_family_score(main_lower, lower)
        score += 3.0 if lower in PRICE_HINTS else 0.0
        score += 1.0 if hint.max_context >= 128_000 else 0.0
        score -= hint.input_price
        candidates.append((score, CheapModelOption(
            model=model_id,
            input_price=hint.input_price,
            output_price=hint.output_price,
            max_context=hint.max_context,
            cross_generation=hint.cross_generation,
            note=hint.note or "discovered by provider model probe",
        )))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return tuple(option for _score, option in candidates[:limit])


def route_from_probe(
    main_model: str,
    models: tuple[ProbeModelInfo, ...] | list[ProbeModelInfo],
    *,
    base_url: str = "",
    provider: str | None = None,
) -> ProbeResult:
    """Build a ModelRoute from provider model inventory when possible."""
    provider_name = provider or infer_provider(base_url, main_model)
    model_ids = tuple(model.id for model in models)
    static_route = find_route(main_model)
    options = discover_cheap_options(main_model, models, provider=provider_name)

    if not options and static_route:
        # If /models is unavailable or omits the cheap model, keep the known route as
        # a safe static fallback; API call/profit guards still decide runtime use.
        options = static_route.cheap_options

    if not options:
        return ProbeResult(
            provider=provider_name,
            main_model=main_model,
            available=False,
            source="models_inventory",
            models_seen=model_ids,
            cheap_options=(),
            route=None,
            failure_reason="no cheap-looking same-provider model discovered",
        )

    main_hint = _price_hint(main_model.lower())
    if static_route:
        main_input = static_route.main_input_price
        main_output = static_route.main_output_price
    else:
        main_input = main_hint.input_price
        main_output = main_hint.output_price
    route = ModelRoute(
        pattern=main_model.lower(),
        main_input_price=main_input,
        main_output_price=main_output,
        cheap_options=options,
    )
    return ProbeResult(
        provider=provider_name,
        main_model=main_model,
        available=True,
        source="models_inventory" if model_ids else "static_fallback",
        models_seen=model_ids,
        cheap_options=options,
        route=route,
    )


class ProviderModelProbe:
    """Safe /models probe for discovering same-platform cheap models."""

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key
        self.timeout = timeout

    def probe(self, main_model: str) -> ProbeResult:
        provider = infer_provider(self.base_url, main_model)
        if not self.base_url or not self.api_key:
            return route_from_probe(main_model, (), base_url=self.base_url, provider=provider)

        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/models", headers=headers)
                response.raise_for_status()
                models = normalize_models_payload(response.json())
        except Exception as e:
            result = route_from_probe(main_model, (), base_url=self.base_url, provider=provider)
            return ProbeResult(
                provider=result.provider,
                main_model=result.main_model,
                available=result.available,
                source="static_fallback_after_probe_failure" if result.available else "probe_failure",
                models_seen=result.models_seen,
                cheap_options=result.cheap_options,
                route=result.route,
                failure_reason=f"/models probe failed: {str(e)[:200]}",
            )
        return route_from_probe(main_model, models, base_url=self.base_url, provider=provider)
