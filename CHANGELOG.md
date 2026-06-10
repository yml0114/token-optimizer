# Changelog

All notable changes to Token Optimizer are documented here.

## [0.2.0] — 2026-06-11

### 🚀 Production-Ready Release

This release turns Token Optimizer from a rule-based compressor into a **production-grade
token cost optimization engine** with profit-aware routing, semantic fidelity guards,
shadow telemetry, and a one-step production entrypoint.

### Added

- **SmartCompressor (v5)** — Profit-aware smart compression pipeline:
  - Rule pre-compression → cheap model intelligent compression → main model reasoning.
  - Same-key zero-config auto-routing: automatically discovers cheap compressor candidates
    from the same API provider (e.g. `mimo-v2.5-pro` → `mimo-v2-flash`).
  - Multi-candidate dynamic selection (`CheapModelOption` + `ModelRoute`).
  - Real/accurate token estimation (tiktoken-first + multilingual fallback).
  - Short-input guard + cheap model context window guard.

- **Dual-Threshold Compression Policy** — Three risk tiers:
  - `safe` (30% target) — standard conversational input.
  - `extreme` (22% target) — high-redundancy, low-risk tool noise.
  - `protected` (45% target) — code, errors, API parameters, file paths.

- **Semantic Fidelity Guard** — `score_semantic_fidelity()` + `FidelityReport`:
  - Zero-cost production guard that blocks lossy compression when critical signals
    (numbers, paths, URLs, emails, code symbols) are dropped.
  - Hard signal coverage scoring with configurable thresholds.
  - Regression corpus: 30/30 safe pass, 30/30 lossy reject.

- **Protected Span Guard** — `extract_protected_spans()` + `format_protected_spans()`:
  - Deterministic extraction of numbers, paths, URLs, emails, code symbols.
  - Injected into the cheap model prompt to enforce verbatim preservation.

- **Self-Learning & Self-Repair** — `CandidateLearningStats`:
  - Tracks success/failure per cheap model candidate.
  - Circuit breaker: stops calling broken compressors after consecutive failures.
  - `safe_passthrough_repair` / `rule_only_self_repair` never let compression break the main path.

- **Shadow Mode / Telemetry** — `SmartCompressor.shadow_evaluate()`:
  - Dry-run v5 routing without calling the cheap model.
  - Records projected savings, policy risk, protected spans, candidate selection, fallback reason.
  - Zero-impact on real request path.

- **ProviderModelProbe** — `ProviderModelProbe`:
  - Safely probes the same provider's `/models` endpoint to discover cheap compressor candidates.
  - Merges probe results with static `ROUTES`; probe failure falls back gracefully.
  - Opt-in via `enable_model_probe=True`.

- **ProductionOptimizer** — One-step production entrypoint:
  - `ProductionOptimizer.chat_completions_create()` wraps the entire pipeline:
    shadow telemetry → rollout gate → smart/rule fallback → prefix reorder →
    cache metadata → HTTP request.
  - `RolloutGate` with four modes: `off` / `shadow` / `auto` / `on`.
  - `auto` mode: only enables real smart compression when shadow predicts positive savings.
  - All optimizer failures fall back to the safest path; the main request is never broken.

- **Adapter-Based Competitor Benchmark** — `benchmark_competitive_adapters.py`:
  - Pluggable adapter framework for official competitor harnesses (LLMLingua, Selective Context, PCToolkit).
  - Safe activation via environment variables; no heavy dependencies by default.

- **SDK Release Artifacts**:
  - `docs/QUICKSTART.md` — 3-step production integration guide.
  - `examples/production_quickstart.py` — real API example.
  - `examples/production_dry_run.py` — local dry-run with mock HTTP.
  - `scripts/production_smoke.py` — zero-dependency smoke test.

### Benchmark Results (v0.2.0)

| Metric | Value |
|--------|-------|
| Unit tests | 120/120 passed |
| Rule-only savings | 31.0% |
| Rule + Flash savings | 72.4% |
| Rule + Flash + Cache savings | 80.5% |
| Fidelity regression (safe) | 30/30 pass |
| Fidelity regression (lossy) | 30/30 reject |
| Shadow mode token saved | 70.4% |
| Shadow mode incremental cost saved | 58.87% |
| Adapter benchmark v5+cache cost saved | 74.1% |
| Adapter benchmark fidelity | 11/11 |

### Changed

- `model_config.py` — Deduplicated `_PROFILES`/`_ALIASES` lookup tables.
- `pyproject.toml` — Development status upgraded from Alpha to Beta.
- Version bumped from `0.1.0` to `0.2.0`.

## [0.1.0] — 2026-06-10

### Initial Release

- L0 Prefix Reordering — deterministic prompt restructuring for cache alignment.
- L2 Prefix Cache — hash-stable prefix tracking for API prefix caching.
- Token cost tracking and reporting.
- OpenAI-compatible `chat.completions.create` interface.
- Support for DeepSeek V4, MiMo V2.5 model families.
