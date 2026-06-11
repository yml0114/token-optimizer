# Adversarial Benchmark Suite

This directory contains adversarial benchmark cases for `token-optimizer`.

Unlike `benchmarks/realworld`, this suite is not designed as a benchmark to overfit.
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
- QA checks: 186/186 (100.0%)
- Perfect cases: 20/20
- Failed cases: 0
- Miss classification: alias_gap=0, true_loss=0, assertion_gap=0

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

## Run

From repository root:

```bash
python3 benchmarks/adversarial/run_adversarial.py
```

Optional minimum pass-rate gate:

```bash
python3 benchmarks/adversarial/run_adversarial.py --min-rate 0.85
```

Strict evaluator gate:

```bash
python3 benchmarks/adversarial/run_adversarial.py --min-rate 1.0
```

The runner writes structured output to:

```text
benchmarks/adversarial/results/latest.json
```

## Evaluator policy

A legacy `qa_groups` entry represents acceptable aliases for the same fact. The benchmark
evaluator therefore treats obvious aliases as OR conditions while still requiring
distinct facts to be present. Supported alias normalization includes:

- numeric formatting aliases such as `12,847` / `12847`;
- decimal-percent aliases such as `0.20` / `20%`;
- simple English number words such as `nine hundred` / `900`;
- same-number same-object phrases such as `30 real-world cases` / `30 cases`.

## Semantic assertion schema

The runner also supports explicit `qa_assertions` for new adversarial cases. If a case
contains `qa_assertions`, the runner uses them directly. Otherwise it converts legacy
`qa_groups` into `any_of` assertions, preserving backward compatibility.

Supported assertion types:

| Type | Meaning | Miss classification |
|---|---|---|
| `present` | one required value or alias group must appear | `true_loss` |
| `any_of` | any value in an alias group may satisfy the fact | `alias_gap` when missed and multiple aliases exist |
| `all_of` | every listed fact must appear independently | `true_loss` |
| `latest_value` | the current/latest corrected value must appear | `true_loss` |
| unsupported type | schema/evaluator cannot interpret the assertion | `assertion_gap` |

Each failed row now includes:

- `miss_details`: per-miss label, assertion type, missing values and classification;
- `miss_summary`: counts for `alias_gap`, `true_loss` and `assertion_gap`.

Do not immediately tune compressor rules just to make this suite green.
Failures should first be classified as one of:

1. real factual-retention weakness (`true_loss`);
2. weak or unsupported assertion design (`assertion_gap`);
3. semantic equivalence not captured by alias matching (`alias_gap`);
4. acceptable compression trade-off.
