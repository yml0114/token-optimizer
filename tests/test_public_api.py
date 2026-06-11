"""Tests for the stable public compression API."""

import pytest

from token_optimizer import CompressionResult, compress_text


def test_compress_text_returns_stable_result_shape():
    text = "Owner Liang. Deadline Jun 25. Budget is 900 USD. Do not retry HTTP 4xx. " * 8

    result = compress_text(
        text,
        mode="safe",
        content_type="memory",
        preserve=["numbers", "dates", "negations", "identifiers"],
    )

    assert isinstance(result, CompressionResult)
    assert result.compressed
    assert result.original_tokens >= result.compressed_tokens
    assert result.mode == "safe"
    assert result.content_type == "memory"
    assert result.method == "adaptive"
    assert result.quality.miss_summary == {"entity_loss": 0, "number_loss": 0, "structure_loss": 0}


def test_compress_text_budget_can_lower_keep_ratio():
    text = "Latency p99 must stay below 200ms. Owner Chen. Region ap-southeast-1. " * 20

    result = compress_text(text, budget_tokens=30, mode="balanced")

    assert result.keep_ratio < 0.75
    assert result.compressed_tokens <= result.original_tokens


def test_compress_text_warns_on_aggressive_high_risk_preserve():
    result = compress_text(
        "Final total is 17.37 USD and invoice id is inv_20260611_8842.",
        mode="aggressive",
        content_type="financial",
        preserve=["numbers", "identifiers"],
    )

    assert result.warnings
    assert result.quality.warnings == result.warnings


def test_compress_text_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        compress_text("hello", mode="unsafe")


def test_compress_text_rejects_non_positive_budget():
    with pytest.raises(ValueError, match="budget_tokens must be positive"):
        compress_text("hello", budget_tokens=0)
