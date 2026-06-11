"""Regression tests for quality benchmark evaluator alias normalization.

These tests protect evaluator behavior added after adversarial benchmark misses
were confirmed to be alias gaps rather than true compression fact loss.
"""

from quality_benchmark import keyword_recall


def assert_recall(compressed: str, keywords: list[str]):
    found, missing = keyword_recall("", compressed, keywords)
    assert found, f"missing aliases: {missing}"
    assert missing == []


def assert_miss(compressed: str, keywords: list[str]):
    found, missing = keyword_recall("", compressed, keywords)
    assert not found
    assert missing


def test_number_word_hundred_alias_matches_numeric_form():
    """'nine hundred' and '900' are aliases, not two independent facts."""
    assert_recall(
        "Token Optimizer gets nine hundred USD. The combined budget is 4.5k dollars.",
        ["nine hundred", "900"],
    )


def test_numeric_form_matches_number_word_hundred_alias():
    """The same alias group should pass when the compressed text uses digits."""
    assert_recall(
        "Token Optimizer gets 900 USD. The combined budget is 4.5k dollars.",
        ["nine hundred", "900"],
    )


def test_same_numbered_object_phrase_alias_matches_compact_phrase():
    """'30 real-world cases' and '30 cases' should be treated as aliases."""
    assert_recall(
        "Need at least 30 real-world cases before claiming production readiness.",
        ["30 real-world cases", "30 cases"],
    )


def test_same_numbered_object_phrase_alias_matches_short_phrase():
    """The shorter same-number same-object phrase should also satisfy the group."""
    assert_recall(
        "Need at least 30 cases before claiming production readiness.",
        ["30 real-world cases", "30 cases"],
    )


def test_distinct_non_alias_facts_are_still_required():
    """Alias normalization must not weaken unrelated AND-style assertions."""
    assert_miss(
        "Need at least 30 real-world cases before claiming production readiness.",
        ["30 real-world cases", "OAuth2"],
    )


def test_same_number_different_object_is_not_alias():
    """Same numbers are not enough when the object/fact type differs."""
    assert_miss(
        "Budget includes 30 credits for the pilot.",
        ["30 credits", "30 cases"],
    )
