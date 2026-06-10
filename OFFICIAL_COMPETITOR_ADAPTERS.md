# Official Competitor Adapter Guide

This project keeps the competitive benchmark safe by default: optional official
competitor packages are detected, but model-backed compression is only executed
when explicitly enabled.

## Why explicit enablement?

Official compressors such as LLMLingua can require heavyweight dependencies,
model downloads, GPU/CPU RAM, or network access. A benchmark or CI run must not
silently download models or fail the production regression suite.

## Current adapter entries

- `LLMLingua-2 official`
- `Selective Context official`
- `PCToolkit official harness`

When dependencies are missing or disabled, benchmark output records
`failure_reason` instead of failing the whole run.

## Run local stable benchmark

```bash
python benchmark_competitive_adapters.py
```

This always runs:

- Raw prompt
- token-optimizer v4 rule-only
- SelectiveContext-like local
- token-optimizer v5 smart-router
- token-optimizer v5 + cache

## Enable official competitors

Set the explicit switch:

```bash
export TOKEN_OPTIMIZER_ENABLE_OFFICIAL_COMPETITORS=1
```

### LLMLingua

Install dependency in an isolated environment:

```bash
pip install llmlingua
```

Then provide a local or already accessible model id/path:

```bash
export TOKEN_OPTIMIZER_LLMLINGUA_MODEL="<local-or-accessible-model>"
export TOKEN_OPTIMIZER_LLMLINGUA_RATE="0.5"
python benchmark_competitive_adapters.py
```

Notes:

- Do not enable this in default CI unless model assets are cached.
- If installation/model loading fails, the benchmark should record an adapter
  failure reason rather than blocking other methods.

### Selective Context

Install the official package or repo according to upstream instructions, then
implement `OfficialSelectiveContextAdapter.compress()` with the stable upstream
API. Keep the output schema unchanged.

### PCToolkit

Use PCToolkit as a benchmark harness only when its package and datasets are
available. The local adapter should map token-optimizer v5 into PCToolkit's
compressor interface without changing production code.

## Output files

- `competitive_adapter_benchmark_results.json`
- `competitive_adapter_benchmark_report.md`

## Safety rules

1. Missing optional dependencies must never fail the whole benchmark.
2. Model-backed official competitors must require explicit env enablement.
3. Every adapter result must include availability, latency, token/cost saving,
   fidelity score, and failure reason when unavailable.
4. Do not compare white-box research methods such as Gisting / 500xCompressor on
   the same executable leaderboard unless model assets and hardware are actually
   available.
