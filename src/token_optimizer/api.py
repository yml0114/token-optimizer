"""Stable public compression API.

This module intentionally stays thin: it exposes a library-friendly interface over
the existing adaptive compressor without turning token-optimizer into a context
manager or memory system.
"""
from __future__ import annotations

from typing import Iterable

from token_optimizer.core.adaptive_compressor import AdaptiveCompressor, ContentType
from token_optimizer.evaluator import evaluate_quality
from token_optimizer.types import CompressionMode, CompressionQuality, CompressionResult

_MODE_KEEP_RATIO: dict[CompressionMode, float] = {
    "safe": 0.95,
    "balanced": 0.75,
    "aggressive": 0.50,
}

_HIGH_RISK_PRESERVE = {"numbers", "dates", "negations", "identifiers", "latest_values"}


def _estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate used by the public API metadata."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def _normalize_mode(mode: str) -> CompressionMode:
    if mode not in _MODE_KEEP_RATIO:
        raise ValueError("mode must be one of: safe, balanced, aggressive")
    return mode  # type: ignore[return-value]


def _risk_warnings(
    *,
    mode: CompressionMode,
    content_type: str | None,
    preserve: Iterable[str] | None,
) -> list[str]:
    warnings: list[str] = []
    preserve_set = {item.lower() for item in preserve or []}
    risky_preserve = bool(_HIGH_RISK_PRESERVE & preserve_set)
    risky_content = content_type in {"code", "json", "financial", "legal", "medical", "memory"}
    if mode == "aggressive" and (risky_preserve or risky_content):
        warnings.append("aggressive mode used with high-risk preservation requirements")
    if preserve_set and mode != "safe" and risky_preserve:
        warnings.append("safe mode is recommended when preserving critical facts")
    return warnings


def compress_text(
    text: str,
    *,
    budget_tokens: int | None = None,
    mode: str = "balanced",
    content_type: str | None = None,
    preserve: list[str] | None = None,
) -> CompressionResult:
    """Compress a text block and return stable metadata.

    Boundaries:
    - This API compresses caller-selected text only.
    - It does not retrieve memory, rank context blocks, or assemble prompts.
    - Quality fields are present so future assertion/evaluator integrations can
      report risk without changing the public return shape.
    """
    normalized_mode = _normalize_mode(mode)
    if budget_tokens is not None and budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive when provided")

    original_tokens = _estimate_tokens(text)
    detected_type = content_type or ContentType.detect(text)
    base_ratio = _MODE_KEEP_RATIO[normalized_mode]
    if budget_tokens is not None and original_tokens > 0:
        budget_ratio = min(1.0, max(0.05, budget_tokens / original_tokens))
        keep_ratio = min(base_ratio, budget_ratio)
    else:
        keep_ratio = base_ratio

    messages = [{"role": "user", "content": text}]
    compressed_messages, stats = AdaptiveCompressor().compress(messages, keep_ratio=keep_ratio)
    compressed = compressed_messages[0].get("content", "") if compressed_messages else ""
    compressed_tokens = _estimate_tokens(compressed)
    warnings = _risk_warnings(mode=normalized_mode, content_type=detected_type, preserve=preserve)
    eval_result = evaluate_quality(text, compressed)
    quality = CompressionQuality(
        passed=eval_result.passed,
        miss_summary=eval_result.to_miss_summary(),
        warnings=warnings,
    )

    return CompressionResult(
        compressed=compressed,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        keep_ratio=stats.get("keep_ratio", keep_ratio),
        method=stats.get("method", "adaptive"),
        mode=normalized_mode,
        content_type=detected_type,
        warnings=warnings,
        quality=quality,
    )
