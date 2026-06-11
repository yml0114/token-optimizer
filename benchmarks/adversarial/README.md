# Adversarial Benchmark Suite

This directory contains adversarial benchmark cases for `token-optimizer`.

Unlike `benchmarks/realworld`, this suite is not designed to stay at 100%.
Its purpose is to expose factual-retention boundaries under harder input conditions:

- paraphrases and synonym wording
- unit and currency variants
- order-shuffled facts
- conflicting updates and corrections
- negation-heavy policies
- noisy long-form meeting/incident notes
- mixed table, bullet and JSON structures
- exact identifiers and comparison thresholds
- multilingual mixed facts

## Latest result

- Cases: 20
- QA checks: 184/186 (98.9%)
- Perfect cases: 18/20
- Failed cases: 2

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

## Run

From repository root:

```bash
python3 benchmarks/adversarial/run_adversarial.py
```

Optional minimum pass-rate gate:

```bash
python3 benchmarks/adversarial/run_adversarial.py --min-rate 0.85
```

The runner writes structured output to:

```text
benchmarks/adversarial/results/latest.json
```

## Policy

Do not immediately tune compressor rules just to make this suite green.
Failures should first be classified as one of:

1. real factual-retention weakness;
2. weak keyword assertion;
3. semantic equivalence not captured by keyword matching;
4. acceptable compression trade-off.
