"""Tests for adversarial benchmark semantic assertion schema."""

from benchmarks.adversarial.run_adversarial import _evaluate_assertion, _qa_groups_to_assertions


def test_legacy_qa_groups_are_converted_to_any_of_assertions():
    case = {"qa_groups": [["nine hundred", "900"], ["Chen"]]}

    assertions = _qa_groups_to_assertions(case)

    assert assertions == [
        {"type": "any_of", "label": "nine hundred/900", "values": ["nine hundred", "900"]},
        {"type": "any_of", "label": "Chen", "values": ["Chen"]},
    ]


def test_any_of_assertion_passes_on_alias():
    found, missing, classification = _evaluate_assertion(
        "Token Optimizer gets 900 USD.",
        {"type": "any_of", "values": ["nine hundred", "900"]},
    )

    assert found
    assert missing == []
    assert classification == "alias_gap"


def test_all_of_assertion_requires_every_fact():
    found, missing, classification = _evaluate_assertion(
        "Chen owns the benchmark.",
        {"type": "all_of", "values": ["Chen", "Jun 25"]},
    )

    assert not found
    assert missing == ["Jun 25"]
    assert classification == "true_loss"


def test_latest_value_assertion_checks_current_value():
    found, missing, classification = _evaluate_assertion(
        "The corrected final benchmark deadline is Jun 25, not Jun 20.",
        {"type": "latest_value", "current": ["Jun 25"], "stale": ["Jun 20"]},
    )

    assert found
    assert missing == []
    assert classification == "true_loss"


def test_unknown_assertion_type_is_classified_as_assertion_gap():
    found, missing, classification = _evaluate_assertion(
        "Chen owns the benchmark.",
        {"type": "custom_rule", "value": "Chen"},
    )

    assert not found
    assert missing == ["unsupported assertion type: custom_rule"]
    assert classification == "assertion_gap"
