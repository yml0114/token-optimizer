# v2.10 Real-world Benchmark Suite

## Summary

`token-optimizer` v2.10 adds a reusable real-world benchmark suite under `benchmarks/realworld/`.

This stage intentionally does **not** continue ratio tuning or compressor rule stacking. The goal is to make quality claims reproducible with realistic samples.

## Latest result

| Metric | Result |
|---|---:|
| Total cases | 30 |
| QA passed | 421/421 |
| QA rate | 100.0% |
| Perfect cases | 30/30 |
| Failed cases | 0 |

## Category breakdown

| Category | Cases | QA |
|---|---:|---:|
| api_json | 5 | 68/68 |
| code_review | 5 | 61/61 |
| incident_postmortem | 5 | 73/73 |
| meeting_notes | 5 | 71/71 |
| product_requirements | 5 | 77/77 |
| technical_specs | 5 | 71/71 |

## Covered scenarios

The suite covers six practical scenario families:

1. API / nested JSON responses
2. Security and correctness code reviews
3. Incident postmortems
4. Meeting notes and multi-project scheduling
5. Product requirements and billing rules
6. Technical specifications and capacity plans

## What this proves

The current quality-first compression path can preserve dense factual details across a broader set of realistic inputs:

- IDs and snake_case fields
- timestamps and dates
- currency and budget figures
- thresholds and comparison conditions
- owners, deadlines and dependencies
- code symbols and security requirements
- nested JSON keys and numeric values

## What this does not prove

This is still not a semantic benchmark and does not prove deep language understanding. It is a reproducible factual-retention benchmark. Future versions should add:

- adversarial paraphrase checks
- longer multi-turn traces
- partial-credit semantic assertions
- independent external datasets

## Validation command

```bash
python3 benchmarks/realworld/run_realworld.py
python3 -m compileall -q benchmarks/realworld/run_realworld.py
```

Latest structured output:

```text
benchmarks/realworld/results/latest.json
```
