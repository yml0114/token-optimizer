# Token Optimizer Competitive Benchmark
Date: 2026-06-11

## Scope
This benchmark compares API-friendly prompt compression paths that can run before the final user model call. White-box methods such as Gisting and 500xCompressor are treated as research upper-bound references, not directly executable black-box API baselines.

## Results
| Method | Tokens | Token Saved | Cost USD | Cost Saved | Fidelity Pass | Avg Fidelity | Avg Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| token-optimizer v5 + cache | 589/2015 | 70.8% | 0.00083320 | 74.1% | 11/11 | 1.0 | 0.035 |
| token-optimizer v5 smart-router | 589/2015 | 70.8% | 0.00120760 | 62.4% | 11/11 | 1.0 | 0.033 |
| SelectiveContext-like local | 1039/2015 | 48.4% | 0.00164800 | 48.7% | 8/11 | 0.8487 | 0.386 |
| token-optimizer v4 rule-only | 1392/2015 | 30.9% | 0.00221700 | 31.0% | 8/11 | 0.8309 | 1.945 |
| Raw prompt | 2015/2015 | 0.0% | 0.00321200 | 0.0% | 11/11 | 1.0 | 0.447 |

## Key Takeaways
- `token-optimizer v5 + cache` ranks first on cost saving in this local reproducible benchmark: **74.1% cost saved**, **70.8% token saved**, **11/11 fidelity pass**.
- `token-optimizer v5 smart-router` without cache still beats local extractive and v4 baselines on cost: **62.4% cost saved** with **11/11 fidelity pass**.
- The local SelectiveContext-like baseline saves **48.7% cost**, but only passes fidelity on **8/11** cases, showing why semantic guards matter.
- v4 rule-only is useful as zero-cost fallback, but saves only **31.0% cost** and fails fidelity on the same number of cases as the extractive baseline in this strict guard.
- Compared with the earlier fixed 30% target benchmark, this competitive benchmark is more conservative because v5 applies the risk-aware protected policy to code/error/number-heavy prompts.

## Benchmark Methods
- Raw prompt: no compression.
- token-optimizer v4 rule-only: deterministic zero-cost fallback.
- SelectiveContext-like local: transparent extractive 50% baseline inspired by self-information context selection; not the official Selective Context package.
- token-optimizer v5 smart-router: rule precompression + risk-aware target + cheap-model cost formula.
- token-optimizer v5 + cache: same as v5 smart-router with 80% prefix-cache assumption, matching `benchmark_l1_v5.py`.

## Next Benchmark Upgrade
Add official adapters for PCToolkit / LLMLingua / Selective Context when dependencies and model assets are available. The current script is the stable local baseline for regression and public comparison.
