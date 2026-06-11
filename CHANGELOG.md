# Changelog

All notable changes to Token Optimizer are documented here.

---

## [2.16.0] — 2026-06-12

### Public API + Semantic Assertion Expansion

#### Added

- **Public compression API** — `compress_text()` stable entry point with three modes:
  - `safe` (95% keep) — critical facts, memory, financial/legal/medical content.
  - `balanced` (75% keep) — default factual compression.
  - `aggressive` (50% keep) — low-risk text with higher loss tolerance.
- **Public result types** — `CompressionResult`, `CompressionQuality`, `CompressionStats`.
  - `miss_summary` with stable keys: `alias_gap`, `true_loss`, `assertion_gap`.
- **API smoke/regression tests** — `tests/test_public_api.py` (5 tests).
- **Semantic assertion adversarial cases** (4 new):
  - `adv_assert_latest_021` — latest corrected value and stale-marker handling.
  - `adv_assert_json_path_022` — nested JSON/path facts and invoice totals.
  - `adv_assert_negation_023` — independent negative safety policies.
  - `adv_assert_threshold_024` — dense threshold comparison conditions.
- **Public API documentation** — `docs/public-api-v2.16.md`.

#### Fixed

- `AdaptiveCompressor` → `NearDeduplicator` constructor: stale `threshold=0.85` → `similarity_threshold=0.85`.

#### Validation

| Metric | Result |
|---|---:|
| Public API tests | 5/5 |
| Adversarial benchmark | 24/24 cases, 205/205 QA, 100.0% |
| Real-world benchmark | 421/421 QA |
| Full pytest | passed |
| Quality benchmark | no regression |
| compileall | passed |

---

## [2.15.0] — 2026-06-11

### Semantic Assertion Diagnostics

#### Added

- **Semantic assertion schema** — `qa_assertions` support for structured quality checks:
  - Types: `present`, `any_of`, `all_of`, `latest_value`.
  - Backward compatible with legacy `qa_groups` (auto-converted to `any_of`).
- **Miss classification** — `miss_details` + `miss_summary` with three categories:
  - `alias_gap` — fact may be preserved but evaluator alias matching insufficient.
  - `true_loss` — genuine fact loss; the only category that should trigger compressor changes.
  - `assertion_gap` — assertion schema or evaluator unable to express/parse the check.
- **Adversarial assertion tests** — `tests/test_adversarial_assertions.py` (5 tests).
- **Updated adversarial docs** — `docs/adversarial-benchmark-v1.md` rewritten for v2.15.
- `.extreme_*.out` added to `.gitignore`.

#### Validation

| Metric | Result |
|---|---:|
| Evaluator tests | 11/11 |
| Adversarial benchmark | 20/20 cases, 186/186 QA, 100.0% |
| Real-world benchmark | 421/421 QA |
| Miss classification | alias_gap=0, true_loss=0, assertion_gap=0 |

---

## [2.14.0] — 2026-06-11

### Evaluator Alias Regression Tests

#### Added

- **Evaluator regression tests** — `tests/test_quality_benchmark_evaluator.py` (6 tests):
  - `nine hundred` ↔ `900` normalization.
  - `30 real-world cases` ↔ `30 cases` same-numbered-object matching.
  - Unrelated facts remain AND; same-number different-object does not falsely merge.
- **Parallel extreme verification** — all test suites run in parallel as smoke gate.

#### Validation

| Metric | Result |
|---|---:|
| Full pytest | 126/126 |
| Adversarial benchmark | 186/186 QA, 100.0% |
| Real-world benchmark | 421/421 QA |
| Quality benchmark | no regression |
| compileall | passed |

---

## [2.13.0] — 2026-06-11

### Evaluator Alias Normalization

#### Changed

- `quality_benchmark.py` evaluator alias normalization:
  - Extended `_normalize_number()` to handle English number words (`<number> hundred`).
  - Added `_same_numbered_object()` for `30 real-world cases` ↔ `30 cases`.
- Compressor unchanged — this release only improves evaluator accuracy.

#### Validation

| Metric | Result |
|---|---:|
| Adversarial benchmark | 186/186 QA, 100.0% |
| Real-world benchmark | 421/421 QA |
| Quality benchmark | no regression |
| compileall | passed |

---

## [2.12.0] — 2026-06-11

### Benchmark CI Workflow

#### Added

- GitHub Actions CI workflow for automated benchmark runs on push/PR.

---

## [2.11.0] — 2026-06-11

### Adversarial Benchmark Suite

#### Added

- **Adversarial benchmark** — `benchmarks/adversarial/` with 20 initial cases covering:
  - Paraphrase, unit variants, order shuffle, conflict updates, negation, long noise,
    structure mixed, code semantics, identifier variants, comparison conditions,
    multilingual noise.
- Automated QA scoring with per-case pass/fail tracking.

---

## [2.10.0] — 2026-06-11

### Real-World Benchmark Suite

#### Added

- **Real-world benchmark** — `benchmarks/realworld/` with 30 cases across:
  - API JSON responses, incident postmortems, code reviews, technical specs, multi-turn conversations.
- Automated fidelity scoring.

---

## [2.9.1] — 2026-06-11

### Real-World Smoke Fixes

#### Fixed

- Edge cases found during real-world benchmark smoke testing.

---

## [2.9.0] — 2026-06-11

### Phase 5c: Fact-Preserving Quality Optimization

#### Changed

- Quality-first compression tuning: density compression and ratio selection calibrated for factual preservation.
- "宁可多留冗余，不可丢失信息" as guiding principle.

---

## [2.8.0] — 2026-06-11

### Phase 5b: Density Compress + Ratio Recalibration

#### Fixed

- `density_compress()` bugs causing over-aggressive compression.
- Ratio selection recalibrated across all content types.

---

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
