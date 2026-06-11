"""Public API result types for token-optimizer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CompressionMode = Literal["safe", "balanced", "aggressive"]


@dataclass(frozen=True)
class CompressionStats:
    """Token and method metadata for a compression call."""

    original_tokens: int
    compressed_tokens: int
    keep_ratio: float
    method: str


@dataclass(frozen=True)
class CompressionQuality:
    """Quality metadata reserved for assertion/evaluator integration."""

    passed: bool | None = None
    miss_summary: dict[str, int] = field(
        default_factory=lambda: {"entity_loss": 0, "number_loss": 0, "structure_loss": 0}
    )
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CompressionResult:
    """Stable public result returned by compress_text()."""

    compressed: str
    original_tokens: int
    compressed_tokens: int
    keep_ratio: float
    method: str
    mode: CompressionMode
    content_type: str | None = None
    warnings: list[str] = field(default_factory=list)
    quality: CompressionQuality = field(default_factory=CompressionQuality)

    @property
    def stats(self) -> CompressionStats:
        return CompressionStats(
            original_tokens=self.original_tokens,
            compressed_tokens=self.compressed_tokens,
            keep_ratio=self.keep_ratio,
            method=self.method,
        )
