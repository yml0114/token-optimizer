"""Lightweight inline quality evaluator for compressed text.

Runs after every compress_text() call to populate CompressionQuality
with real entity recall, number precision, and structural integrity scores
instead of the previous None placeholder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalResult:
    entity_recall: float
    number_precision: float
    structural_preservation: float
    overall_score: float
    passed: bool | None

    def to_miss_summary(self) -> dict[str, int]:
        return {
            "entity_loss": round((1 - self.entity_recall) * 100),
            "number_loss": round((1 - self.number_precision) * 100),
            "structure_loss": round((1 - self.structural_preservation) * 100),
        }


_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-zA-Z]+\b|\b\d[\d,.]*\b|\b[a-zA-Z]+\d+\b")
_NUMBER_PATTERN = re.compile(r"\b\d[\d,.]*\b")


def entity_recall(original: str, compressed: str) -> float:
    """Fraction of original entities (capitalized words, alphanumeric IDs)
    preserved in the compressed output."""
    orig = set(_ENTITY_PATTERN.findall(original))
    comp = set(_ENTITY_PATTERN.findall(compressed))
    if not orig:
        return 1.0
    return len(orig & comp) / len(orig)


def number_precision(original: str, compressed: str) -> float:
    """Fraction of original number tokens preserved exactly."""
    orig = set(_NUMBER_PATTERN.findall(original))
    comp = set(_NUMBER_PATTERN.findall(compressed))
    if not orig:
        return 1.0
    return len(orig & comp) / len(orig)


def structural_preservation(original: str, compressed: str) -> float:
    """How well JSON/structural markers ({}, [], :") survive compression."""
    markers = ["{", "}", "[", "]", '":', "':"]
    scores: list[float] = []
    for ch in markers:
        oc = original.count(ch)
        if oc == 0:
            continue
        scores.append(min(compressed.count(ch), oc) / oc)
    if not scores:
        return 1.0  # no structural markers in original — nothing to lose
    return sum(scores) / len(scores)


def evaluate_quality(
    original: str,
    compressed: str,
    *,
    pass_threshold: float = 0.70,
) -> EvalResult:
    """Run all quality checks and return a structured result.

    Returns passed=None when original is empty (nothing to evaluate).
    """
    if not original.strip():
        return EvalResult(
            entity_recall=1.0,
            number_precision=1.0,
            structural_preservation=1.0,
            overall_score=1.0,
            passed=None,
        )

    er = entity_recall(original, compressed)
    np_ = number_precision(original, compressed)
    sp = structural_preservation(original, compressed)

    overall = 0.40 * er + 0.35 * np_ + 0.25 * sp
    passed = overall >= pass_threshold

    return EvalResult(
        entity_recall=er,
        number_precision=np_,
        structural_preservation=sp,
        overall_score=overall,
        passed=passed,
    )
