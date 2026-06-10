# Adapter-based Competitive Benchmark
Date: 2026-06-11

## Scope
This benchmark uses a stable compressor adapter schema. Local baselines run now; official competitor packages are recorded as unavailable when dependencies or model assets are missing.

## Results
| Method | Available | Tokens | Token Saved | Cost USD | Cost Saved | Fidelity Pass | Avg Fidelity | Avg Latency ms | Failure Reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| token-optimizer v5 + cache | yes | 589/2015 | 70.8% | 0.00083320 | 74.1% | 11/11 | 1.0 | 1.33 | - |
| token-optimizer v5 smart-router | yes | 589/2015 | 70.8% | 0.00120760 | 62.4% | 11/11 | 1.0 | 1.365 | - |
| SelectiveContext-like local | yes | 1039/2015 | 48.4% | 0.00164800 | 48.7% | 8/11 | 0.8487 | 0.19 | - |
| token-optimizer v4 rule-only | yes | 1392/2015 | 30.9% | 0.00221700 | 31.0% | 8/11 | 0.8309 | 1.727 | - |
| Raw prompt | yes | 2015/2015 | 0.0% | 0.00321200 | 0.0% | 11/11 | 1.0 | 0.001 | - |
| Selective Context official | no | - | - | - | - | - | - | - | Python package 'selective_context' is not installed |
| LLMLingua-2 official | no | - | - | - | - | - | - | - | Python package 'llmlingua' is not installed |
| PCToolkit official harness | no | - | - | - | - | - | - | - | PCToolkit package is not installed |

## Key Takeaways
- token-optimizer v5 + cache remains the strongest available local/API-friendly baseline on cost saving and fidelity.
- Optional official adapters for LLMLingua-2, Selective Context and PCToolkit are now first-class benchmark entries instead of TODO notes.
- Missing competitor dependencies no longer block regression; they are recorded with explicit failure reasons.
- Gisting and 500xCompressor should remain white-box research upper-bound references unless model assets and hardware are available.

## Next Step
Install official competitor dependencies in an isolated environment, then fill each adapter body while preserving this output schema.
