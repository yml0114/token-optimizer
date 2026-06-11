# v2.15 Semantic Assertion Schema

## Summary

`token-optimizer` v2.15 upgrades the adversarial benchmark evaluator output from a plain keyword-group checker into a minimal semantic assertion schema.

This release still does **not** change the compressor. It changes the adversarial runner so future failures can be diagnosed before anyone tunes compression behavior.

## What changed

`benchmarks/adversarial/run_adversarial.py` now supports:

1. Backward-compatible legacy `qa_groups` conversion.
2. New explicit `qa_assertions` schema for future adversarial cases.
3. Per-miss diagnostic fields:
   - `miss_details`
   - `miss_summary`
4. Miss classification buckets:
   - `alias_gap`
   - `true_loss`
   - `assertion_gap`

Existing adversarial cases do not need to be rewritten. If a case only has `qa_groups`, each group is converted into an `any_of` assertion.

## Supported assertion types

| Type | Meaning | Miss classification |
|---|---|---|
| `present` | one required value or alias group must appear | `true_loss` |
| `any_of` | any value in an alias group may satisfy the fact | `alias_gap` when missed and multiple aliases exist |
| `all_of` | every listed fact must appear independently | `true_loss` |
| `latest_value` | the current/latest corrected value must appear | `true_loss` |
| unsupported type | schema/evaluator cannot interpret the assertion | `assertion_gap` |

## Latest adversarial result

| Metric | Result |
|---|---:|
| Total cases | 20 |
| QA passed | 186/186 |
| QA rate | 100.0% |
| Perfect cases | 20/20 |
| Failed cases | 0 |
| alias_gap | 0 |
| true_loss | 0 |
| assertion_gap | 0 |

## Why this matters

Before v2.15, a miss was only a string such as:

```text
30 real-world cases/30 cases
```

That was not enough to know whether the system had:

1. truly lost a fact;
2. preserved the fact but used an unrecognized alias;
3. exposed a weak or unsupported assertion design.

v2.15 makes this distinction explicit in the result JSON. That keeps future work honest: compressor changes should be driven by `true_loss`, not by `alias_gap` or `assertion_gap`.

## Validation commands

```bash
PYTHONPATH=src:. python3 -m pytest tests/test_adversarial_assertions.py tests/test_quality_benchmark_evaluator.py
python3 benchmarks/adversarial/run_adversarial.py --min-rate 1.0
python3 benchmarks/realworld/run_realworld.py
PYTHONPATH=src python3 -m pytest
python3 quality_benchmark.py
python3 -m compileall -q src quality_benchmark.py benchmarks tests
```

Latest structured output:

```text
benchmarks/adversarial/results/latest.json
```

## What did not change

The compressor was not modified in this release. v2.15 is strictly an evaluator and benchmark-diagnostics improvement.
