<p align="center">
  <strong>Token Optimizer</strong>
</p>

<p align="center">
  <strong>Production-Grade LLM Token Cost Optimization Engine</strong><br>
  Save 70-80% on API costs. Zero quality loss. One-line integration.
</p>

<p align="center">
  <code>pip install token-optimizer</code>
</p>

---

## What is Token Optimizer?

Token Optimizer is a **production-ready token cost optimizer** that sits between your application and any OpenAI-compatible LLM API. It compresses input tokens aggressively while preserving all critical task signals — numbers, paths, URLs, emails, code symbols, and constraints — so your main model gets cheaper input without losing accuracy.

**Core principle:** The cheap model only compresses. Your main model still does the thinking.

### How it saves money

```
User request (10,000 tokens)
    │
    ▼
┌─────────────────────────────────────┐
│  Rule Pre-Compression (L1 v4)       │  ← Zero-cost filler/noise stripping
│  10,000 → 6,900 tokens              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Profit-Aware Smart Router          │  ← Predicts: is calling a cheap model worth it?
│  "mimo-v2-flash can save 68%?"      │
└─────────────────────────────────────┘
    │  (only if predicted savings > threshold)
    ▼
┌─────────────────────────────────────┐
│  Cheap Model Compression            │  ← e.g. mimo-v2-flash ($0.10/M) compresses
│  6,900 → 2,000 tokens               │     for mimo-v2.5-pro ($1.00/M)
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Semantic Fidelity Guard             │  ← Blocks if critical signals were dropped
│  score: 0.92 ≥ 0.75 threshold ✓     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Your Main Model                    │  ← Receives 2,000 tokens instead of 10,000
│  (unchanged, same API key)          │     Saves ~80% on input cost
└─────────────────────────────────────┘
```

---

## Quick Start (3 steps)

### 1. Install

```bash
pip install token-optimizer
```

### 2. Integrate

```python
from token_optimizer import ProductionOptimizer, ProductionOptimizerConfig, RolloutGate

optimizer = ProductionOptimizer(ProductionOptimizerConfig(
    model="mimo-v2.5-pro",          # your main model
    api_key="your-api-key",
    base_url="https://api.xiaomimimo.com/v1",
    rollout=RolloutGate(mode="auto"),  # shadow → auto → on
))

# Use exactly like OpenAI SDK
response = optimizer.chat_completions_create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Analyze this 10K token document..."},
    ],
)

# Check what the optimizer did
print(response["_optimization"]["token_saved_pct_est"])  # e.g. 78.5
print(response["_optimization"]["l1_smart_compression"]["mode"])  # "smart"
```

### 3. Roll out safely

```
Week 1:  mode="shadow"   → Observe telemetry, zero risk
Week 2:  mode="auto"     → Auto-enable when shadow predicts savings
Week 3:  mode="on"       → Force-enable after confidence is built
```

---

## 6-Layer Safety Architecture

Token Optimizer is designed so that **no optimizer failure can ever break your main request** or increase your costs.

| # | Safety Layer | What it does | Failure mode |
|---|---|---|---|
| 1 | **Profit-Aware Routing** | Only calls cheap model when predicted savings exceed threshold | Falls back to rule-only (free) |
| 2 | **Short-Input Guard** | Skips cheap model for tiny requests where fixed cost isn't worth it | Returns rule-compressed result |
| 3 | **Context Window Guard** | Checks if input fits cheap model's context limit | Falls back to rule-only |
| 4 | **Semantic Fidelity Guard** | Blocks lossy compression when critical signals (numbers, paths, URLs, emails, code) are dropped | Returns rule-compressed result |
| 5 | **Circuit Breaker** | Stops calling broken cheap models after consecutive failures | Falls back to rule-only, auto-recovers |
| 6 | **Safe Passthrough** | Any uncaught exception → return original messages untouched | Main request proceeds unmodified |

**Shadow Mode** lets you observe what the optimizer *would* do without actually calling the cheap model — zero risk during rollout.

---

## Supported Models

| Model | Role | Input $/M | Output $/M | Cache $/M |
|-------|------|-----------|------------|-----------|
| `mimo-v2.5-pro` | Main | $1.00 | $3.00 | $0.20 |
| `mimo-v2.5` | Main / Cheap | $0.14 | $0.28 | $0.0028 |
| `mimo-v2-flash` | Cheap | $0.10 | $0.30 | $0.01 |
| `deepseek-v4-flash` | Main / Cheap | $0.14 | $0.28 | $0.0028 |
| `deepseek-v4-pro` | Main | $0.435 | $0.87 | $0.003625 |
| `gpt-4o` → `gpt-4o-mini` | Auto-routed | — | — | — |
| `claude-3-opus` → `claude-3-haiku` | Auto-routed | — | — | — |
| `qwen-max` → `qwen-turbo` | Auto-routed | — | — | — |

Auto-routing works for any model with a known cheap sibling. Custom routes can be added via `ModelRoute`.

---

## Benchmark Results

All benchmarks are runnable locally (`python benchmark_*.py`). Results from v0.2.0:

### Cost Savings

| Configuration | Input Cost Saved | Notes |
|---|---|---|
| Rule-only (L1 v4) | **31.0%** | Zero API cost, deterministic |
| Rule + Cheap Model | **72.4%** | Profit-aware smart routing |
| Rule + Cheap Model + Cache | **80.5%** | With prefix cache alignment |

### Quality Guarantees

| Test | Result |
|---|---|
| Fidelity regression (safe inputs) | **30/30 pass** |
| Fidelity regression (lossy inputs) | **30/30 reject** |
| Adapter benchmark fidelity | **11/11 pass** |

### Shadow Mode Telemetry

| Metric | Value |
|---|---|
| Token saved (rule → smart) | 70.4% |
| Incremental cost saved | 58.87% |

Run benchmarks yourself:

```bash
python benchmark_l1_v5.py               # core savings benchmark
python benchmark_fidelity_regression.py  # quality regression
python benchmark_shadow_mode.py          # shadow telemetry
python benchmark_competitive_adapters.py # competitor comparison
python scripts/production_smoke.py       # smoke test (no API needed)
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  ProductionOptimizer                  │
│  .chat_completions_create(messages=[...])             │
│                                                      │
│  ┌─────────┐   ┌──────────┐   ┌──────────────────┐  │
│  │ Shadow   │→ │ Rollout   │→ │ Smart Compressor  │  │
│  │ Telemetry│   │ Gate      │   │ (L1 v5)          │  │
│  └─────────┘   └──────────┘   └──────────────────┘  │
│       │                            │                 │
│       ▼                            ▼                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │ Provider  │   │ Prefix   │   │ HTTP Request     │ │
│  │ Model     │   │ Reorder  │   │ (OpenAI-compat)  │ │
│  │ Probe     │   │ + Cache  │   │                  │ │
│  └──────────┘   └──────────┘   └──────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### Core Modules

| Module | File | Responsibility |
|---|---|---|
| `SmartCompressor` | `core/smart_compressor.py` | Profit-aware compression with fidelity guards |
| `InputCompressor` | `core/signal_noise.py` | Rule-based L1 v4: signal/noise classification, filler stripping, history compression |
| `ProviderModelProbe` | `core/model_probe.py` | Discovers cheap compressor candidates from provider's `/models` |
| `ProductionOptimizer` | `production.py` | One-step production entrypoint |
| `ModelProfile` | `models/model_config.py` | Model pricing and cache configuration |

---

## Configuration Reference

### ProductionOptimizerConfig

```python
ProductionOptimizerConfig(
    model="mimo-v2.5-pro",           # your main model
    api_key="sk-...",                # same key for main + cheap model
    base_url="https://api.xiaomimimo.com/v1",
    rollout=RolloutGate(mode="auto"), # off / shadow / auto / on
    enable_model_probe=True,         # probe /models for cheap candidates
    enable_prefix_reorder=True,      # deterministic prefix for cache alignment
    enable_prefix_cache=True,        # track prefix hash for cache hits
    request_timeout=120.0,
    smart_min_rule_tokens=128,       # minimum tokens to justify cheap model call
    attach_metadata=True,            # attach _optimization to response
)
```

### RolloutGate

```python
RolloutGate(
    mode="auto",                     # off / shadow / auto / on
    min_estimated_savings_pct=20.0,  # minimum predicted savings to enable
    allow_protected=True,            # allow smart compression on protected inputs
    max_protected_span_count=64,     # max protected spans before blocking
    require_probe_available=False,   # require /models probe to succeed
)
```

### SmartCompressor (advanced)

```python
SmartCompressor(
    main_model="mimo-v2.5-pro",
    api_key="sk-...",
    base_url="https://api.xiaomimimo.com/v1",
    level=CompressionLevel.AGGRESSIVE,
    min_profit_margin=0.15,          # minimum predicted savings ratio
    expected_smart_ratio=0.30,       # target compression ratio
    safe_target_ratio=0.30,          # safe policy target
    extreme_target_ratio=0.22,       # extreme policy target
    protected_target_ratio=0.45,     # protected policy target
    learning_enabled=True,           # self-learning from success/failure
    max_consecutive_failures=2,      # circuit breaker threshold
    circuit_breaker_cooldown=20,     # calls before retry
    enable_model_probe=False,        # probe /models endpoint
)
```

---

## API Reference

### `ProductionOptimizer.chat_completions_create(**kwargs)`

OpenAI-compatible call that adds optimization on top.

**Returns:** Standard OpenAI response dict with `_optimization` key attached:

```python
{
    "choices": [...],
    "usage": {...},
    "_optimization": {
        "production_optimizer": True,
        "model": "mimo-v2.5-pro",
        "rollout_mode": "auto",
        "token_saved_pct_est": 78.5,
        "input_tokens_original_est": 10000,
        "input_tokens_optimized_est": 2150,
        "l1_smart_compression": {
            "mode": "smart",
            "compressor": "mimo-v2-flash",
            "fidelity_guard": {"score": 0.92, "passed": True},
            "profit_guard": {"actual": {"savings_pct": 72.4}},
        },
        "prefix_reorder": {...},
        "cache_stable": True,
        "cache_savings_estimate": {...},
        "rollout_gate": {"enabled": True, "reason": "auto_gate_passed"},
        "shadow": {...},
        "errors": [],
        "latency_ms": 1450.32,
    }
}
```

### `SmartCompressor.compress(messages, system_text="")`

Direct compression call (used internally by ProductionOptimizer).

**Returns:** `(compressed_messages, metadata_dict)`

### `SmartCompressor.shadow_evaluate(messages, system_text="")`

Dry-run that records what the optimizer *would* do without calling the cheap model.

**Returns:** Telemetry dict with `would_call_smart`, `estimated_savings_pct`, `selected_candidate`, etc.

### `ProviderModelProbe(base_url, api_key, timeout=10).probe(main_model)`

Probe the provider's `/models` endpoint for cheap compressor candidates.

**Returns:** `ProbeResult` with `route`, `available`, `source`, `models_seen`, `failure_reason`.

---

## Project Structure

```
token-optimizer/
├── src/token_optimizer/
│   ├── __init__.py              # exports
│   ├── production.py            # ProductionOptimizer (main entrypoint)
│   ├── config.py                # OptimizerConfig
│   ├── client.py                # TokenOptimizer (legacy)
│   ├── core/
│   │   ├── smart_compressor.py  # SmartCompressor, fidelity, policy, routing
│   │   ├── signal_noise.py      # L1 v4 rule compressor, signal/noise classifier
│   │   ├── model_probe.py       # ProviderModelProbe
│   │   └── prompt_reorderer.py  # L0 prefix reordering
│   └── models/
│       ├── model_config.py      # ModelProfile, pricing, cache config
│       └── mimo_adapter.py      # MiMo-specific API normalization
├── tests/                       # 120 unit tests
├── examples/
│   ├── production_quickstart.py # real API example
│   └── production_dry_run.py    # mock HTTP example
├── scripts/
│   └── production_smoke.py      # zero-dependency smoke test
├── docs/
│   └── QUICKSTART.md            # integration guide
├── benchmark_l1_v5.py           # core savings benchmark
├── benchmark_fidelity_regression.py
├── benchmark_shadow_mode.py
├── benchmark_competitive.py
└── benchmark_competitive_adapters.py
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest -q              # 120 tests

# Run smoke test (no API key needed)
python scripts/production_smoke.py

# Run all benchmarks
python benchmark_l1_v5.py
python benchmark_fidelity_regression.py
python benchmark_shadow_mode.py
python benchmark_competitive_adapters.py

# Lint
ruff check src/ tests/
```

---

## Competitive Positioning

Token Optimizer vs. existing tools:

| Feature | Token Optimizer | LLMLingua | Selective Context | Gisting |
|---|---|---|---|---|
| Zero-config auto-routing | ✅ | ❌ | ❌ | ❌ |
| Profit-aware routing | ✅ | ❌ | ❌ | ❌ |
| Semantic fidelity guard | ✅ | ❌ | ❌ | ❌ |
| Protected span guard | ✅ | ❌ | ❌ | ❌ |
| Shadow mode telemetry | ✅ | ❌ | ❌ | ❌ |
| Self-learning circuit breaker | ✅ | ❌ | ❌ | ❌ |
| Production rollout gate | ✅ | ❌ | ❌ | ❌ |
| Prefix cache optimization | ✅ | ❌ | ❌ | ❌ |
| No model download required | ✅ | ❌ | ✅ | ❌ |
| OpenAI-compatible wrapper | ✅ | ❌ | ❌ | ❌ |
| Cost: rule-only path | Free | — | — | — |
| Cost: smart path | ~$0.0001/req | — | — | — |

---

## License

MIT
