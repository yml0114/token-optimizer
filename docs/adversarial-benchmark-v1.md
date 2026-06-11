# v2.11 Adversarial Benchmark Suite

## Summary

`token-optimizer` v2.11 adds a dedicated adversarial benchmark suite under `benchmarks/adversarial/`.

This stage intentionally does **not** modify the compressor. The purpose is to measure robustness against harder factual-preservation cases after v2.10 established a 30-case real-world benchmark suite.

## Latest result

| Metric | Result |
|---|---:|
| Total cases | 20 |
| QA passed | 184/186 |
| QA rate | 98.9% |
| Perfect cases | 18/20 |
| Failed cases | 2 |

## Category breakdown

| Category | Cases | QA | Failed cases |
|---|---:|---:|---:|
| code_semantics | 2 | 27/27 | 0 |
| comparison_conditions | 1 | 14/14 | 0 |
| conflict_update | 2 | 12/12 | 0 |
| identifier_variants | 1 | 6/6 | 0 |
| long_noise | 2 | 19/20 | 1 |
| multilingual_noise | 1 | 10/10 | 0 |
| negation | 2 | 17/17 | 0 |
| order_shuffle | 2 | 19/19 | 0 |
| paraphrase | 2 | 14/14 | 0 |
| structure_mixed | 2 | 23/23 | 0 |
| unit_variants | 3 | 23/24 | 1 |

## Failed / boundary cases

| Case | Category | QA | Misses |
|---|---|---:|---|
| adv_noise_meeting_011 | long_noise | 7/8 | 30 real-world cases/30 cases |
| adv_units_budget_003 | unit_variants | 6/7 | nine hundred/900 |

## Interpretation

The adversarial result is intentionally not treated as a failure. A 98.9% QA rate with 18/20 perfect cases means the current quality-first compressor is robust on most adversarial factual cases, while still exposing useful boundary conditions.

Current boundary observations:

1. `adv_noise_meeting_011` missed the `30 real-world cases` dependency in a noisy meeting note.
2. `adv_units_budget_003` missed the Token Optimizer budget represented as `nine hundred USD` / `900`.

These are useful future optimization targets, but they should not be fixed by blindly adding benchmark-specific rules.

## What this proves

The v2.9.1/v2.10 fact-preserving path is resilient across:

- paraphrased business fields;
- exact identifiers and snake_case / camelCase variants;
- negation-heavy refund and balance policies;
- comparison thresholds;
- mixed Markdown tables, bullets and JSON;
- multilingual Chinese/English facts;
- code-symbol preservation.

## What this does not prove

This is still keyword-based factual retention, not deep semantic validation. Future work should add semantic assertions for:

- latest-value correctness in conflict updates;
- number-word equivalence such as `nine hundred` ↔ `900`;
- final-state extraction in noisy long multi-turn traces;
- contradiction and negation reasoning.

## Validation commands

```bash
python3 benchmarks/adversarial/run_adversarial.py
python3 benchmarks/adversarial/run_adversarial.py --min-rate 0.85
python3 benchmarks/realworld/run_realworld.py
python3 quality_benchmark.py
python3 -m compileall -q src quality_benchmark.py benchmarks tests
```

Latest structured output:

```text
benchmarks/adversarial/results/latest.json
```
