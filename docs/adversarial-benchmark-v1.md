# v2.13 Evaluator Alias Normalization

## Summary

`token-optimizer` v2.13 improves the benchmark evaluator, not the compressor.

The v2.11 adversarial suite originally reported two missed QA checks. Manual inspection showed both facts were preserved in the compressed output:

1. `adv_units_budget_003` preserved `nine hundred USD`, but the evaluator did not treat it as equivalent to `900`.
2. `adv_noise_meeting_011` preserved `30 real-world cases`, but the evaluator did not treat it as equivalent to `30 cases`.

v2.13 fixes these evaluator alias gaps in `quality_benchmark.py`.

## Latest adversarial result

| Metric | Result |
|---|---:|
| Total cases | 20 |
| QA passed | 186/186 |
| QA rate | 100.0% |
| Perfect cases | 20/20 |
| Failed cases | 0 |

## Category breakdown

| Category | Cases | QA | Failed cases |
|---|---:|---:|---:|
| code_semantics | 2 | 27/27 | 0 |
| comparison_conditions | 1 | 14/14 | 0 |
| conflict_update | 2 | 12/12 | 0 |
| identifier_variants | 1 | 6/6 | 0 |
| long_noise | 2 | 20/20 | 0 |
| multilingual_noise | 1 | 10/10 | 0 |
| negation | 2 | 17/17 | 0 |
| order_shuffle | 2 | 19/19 | 0 |
| paraphrase | 2 | 14/14 | 0 |
| structure_mixed | 2 | 23/23 | 0 |
| unit_variants | 3 | 24/24 | 0 |

## Failed / boundary cases

| Case | Category | QA | Misses |
|---|---|---:|---|
| none | - | - | - |

## What changed

The evaluator now recognizes:

- simple English number words: `zero` through `ninety`, plus `<number> hundred`;
- same-number same-object aliases such as `30 real-world cases` and `30 cases`.

This keeps the benchmark honest: the compressor is not rewarded for losing facts,
but it is also not penalized when it preserves a fact using an equivalent surface form.

## What did not change

The compressor was not modified in this release. v2.13 is strictly an evaluator quality improvement.

## Validation commands

```bash
python3 benchmarks/adversarial/run_adversarial.py --min-rate 1.0
python3 benchmarks/realworld/run_realworld.py
python3 quality_benchmark.py
python3 -m compileall -q src quality_benchmark.py benchmarks tests
```

Latest structured output:

```text
benchmarks/adversarial/results/latest.json
```
