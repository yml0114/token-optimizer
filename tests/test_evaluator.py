"""Tests for token_optimizer.evaluator — inline quality checks."""

import pytest

from token_optimizer.evaluator import (
    EvalResult,
    entity_recall,
    number_precision,
    structural_preservation,
    evaluate_quality,
)


class TestEntityRecall:
    def test_full_recall(self):
        assert entity_recall("Alice met Bob at 42 Main St.", "Alice met Bob at 42 Main St.") == 1.0

    def test_partial_recall(self):
        er = entity_recall("Alice met Bob and Charlie.", "Alice met Bob.")
        assert er == 2 / 3

    def test_empty_original(self):
        assert entity_recall("", "anything") == 1.0

    def test_numeric_id_preserved(self):
        assert entity_recall("User id12847 logged in.", "User id12847.") == 1.0


class TestNumberPrecision:
    def test_all_preserved(self):
        assert number_precision("Price: $45.99, Qty: 12", "Price: $45.99, Qty: 12") == 1.0

    def test_missing_number(self):
        np_ = number_precision("a=5, b=10, c=15", "a=5, c=15")
        assert np_ == 2 / 3

    def test_empty_original(self):
        assert number_precision("", "42") == 1.0


class TestStructuralPreservation:
    def test_json_intact(self):
        sp = structural_preservation('{"key": "value"}', '{"key": "value"}')
        assert sp == 1.0

    def test_json_collapsed(self):
        sp = structural_preservation('{"a": 1, "b": 2}', "some summary")
        assert sp < 1.0

    def test_empty_original(self):
        assert structural_preservation("", "{}") == 1.0


class TestEvaluateQuality:
    def test_identical_passes(self):
        result = evaluate_quality("Alice has $45.", "Alice has $45.")
        assert result.passed is True
        assert result.overall_score >= 0.99
        assert result.entity_recall == 1.0
        assert result.number_precision == 1.0
        assert result.structural_preservation == 1.0

    def test_empty_original_returns_none(self):
        result = evaluate_quality("", "compressed")
        assert result.passed is None
        assert result.overall_score == 1.0

    def test_heavily_compressed_fails_threshold(self):
        result = evaluate_quality(
            "Alice and Bob went to 42 Main St. and spent $45 each.",
            "two people spent money",
        )
        assert result.passed is False
        assert result.overall_score < 0.70

    def test_miss_summary(self):
        result = evaluate_quality("Alice $45 Bob $10", "Alice")
        summary = result.to_miss_summary()
        assert summary["entity_loss"] > 0
        assert summary["number_loss"] > 0


class TestEvalResult:
    def test_to_miss_summary_perfect(self):
        er = EvalResult(1.0, 1.0, 1.0, 1.0, True)
        assert er.to_miss_summary() == {
            "entity_loss": 0,
            "number_loss": 0,
            "structure_loss": 0,
        }

    def test_to_miss_summary_partial(self):
        er = EvalResult(0.5, 0.8, 0.9, 0.73, True)
        s = er.to_miss_summary()
        assert s["entity_loss"] == 50
        assert s["number_loss"] == 20
        assert s["structure_loss"] == 10
