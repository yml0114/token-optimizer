# Real-world Benchmark Suite

This directory contains reusable real-world benchmark cases for `token-optimizer`.

## Purpose

The benchmark is designed to test factual preservation on practical, high-density inputs instead of only tuning synthetic compression ratios.

Current suite:

- Cases: 30
- QA checks: 421/421 (100.0%)
- Perfect cases: 30/30
- Failed cases: 0

## Categories

| Category | Cases | QA |
|---|---:|---:|
| api_json | 5 | 68/68 |
| code_review | 5 | 61/61 |
| incident_postmortem | 5 | 73/73 |
| meeting_notes | 5 | 71/71 |
| product_requirements | 5 | 77/77 |
| technical_specs | 5 | 71/71 |

## Case format

Each case is a JSON file under `benchmarks/realworld/cases/`.

Required fields:

- `id`: stable case identifier.
- `category`: scenario category.
- `title`: human-readable title.
- `messages`: chat messages passed to the compressor.
- `qa_groups`: list of acceptable keyword groups. A group passes if any keyword in that group is preserved.

Optional fields:

- `must_keep`: semantic description of important facts.
- `notes`: test intent.

## Run

From repository root:

```bash
python3 benchmarks/realworld/run_realworld.py
```

The runner writes structured results to:

```text
benchmarks/realworld/results/latest.json
```

## Policy

This suite should grow with real cases. Do not tune rules only to satisfy this benchmark. If a case fails, first decide whether it exposes a real product-quality issue or a weak QA assertion.
