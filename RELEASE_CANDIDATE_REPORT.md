# Token Optimizer Release Candidate Report

本轮把项目从“生产入口已完成”继续补成 SDK 发布形态。

## 新增交付物

- `docs/QUICKSTART.md`
- `examples/production_quickstart.py`
- `examples/production_dry_run.py`
- `scripts/production_smoke.py`

## SDK 接入方式

```python
from token_optimizer import ProductionOptimizer, ProductionOptimizerConfig, RolloutGate

optimizer = ProductionOptimizer(ProductionOptimizerConfig(
    model="mimo-v2.5-pro",
    api_key="sk-...",
    base_url="https://api.xiaomimimo.com/v1",
    rollout=RolloutGate(mode="auto", min_estimated_savings_pct=20),
    enable_model_probe=True,
))

response = optimizer.chat_completions_create(
    messages=[{"role": "user", "content": "..."}],
)

print(response["_optimization"])
```

## 本地 Smoke Test

```bash
python scripts/production_smoke.py
```

该脚本不调用外部 API，通过继承 `ProductionOptimizer` mock `_sync_request()` 验证 SDK surface 和 `_optimization` metadata contract。

## 上线建议

1. `shadow`：只观测，不调用 cheap model。
2. `auto`：收益和风险通过门禁才启用。
3. `on`：强制尝试 smart compression，但所有 guard 仍生效。

## 当前验证

```text
python scripts/production_smoke.py ✅
python -m pytest -q ✅
python benchmark_fidelity_regression.py ✅
python benchmark_l1_v5.py ✅
python benchmark_competitive_adapters.py ✅
python benchmark_shadow_mode.py ✅
git diff --check ✅
```

## 当前能力闭环

```text
ProductionOptimizer
+ ProviderModelProbe
+ Shadow Mode / Telemetry
+ RolloutGate
+ Profit-Aware Smart Router
+ Protected Span Guard
+ Semantic Fidelity Guard
+ Context Guard
+ Circuit Breaker / Self Repair
+ Prefix Reorder / Cache Metadata
+ Competitive Benchmarks
+ Fidelity Regression Corpus
```
