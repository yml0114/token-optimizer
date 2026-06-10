# Production Optimizer Report

本轮把 Token Optimizer 收口成可业务直接接入的生产成品：`ProductionOptimizer`。

## 目标

不再让业务方自己拼 smart compression、shadow telemetry、rollout gate、provider probe、prefix reorder、cache metadata、HTTP 请求和异常兜底，而是提供一条生产链路：

```text
输入 messages
→ strip dynamic fields
→ shadow_evaluate 预测收益和风险
→ rollout gate 决定是否真实启用 smart compression
→ smart / rule-only / passthrough safe fallback
→ prefix reorder
→ 主模型 chat completions
→ 返回响应 + _optimization telemetry
```

## 新增文件

- `src/token_optimizer/production.py`
- `tests/test_production_optimizer.py`

## 对外 API

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

## Rollout Modes

| mode | 行为 |
|---|---|
| `off` | 不启用 smart compression，只做安全路径 |
| `shadow` | 只旁路观测，不真实调用 cheap model |
| `auto` | 根据 shadow telemetry 自动门禁，收益/风险达标才启用 |
| `on` | 强制尝试 smart compression，但仍保留所有 guard 和 fallback |

## 安全保证

- shadow 异常不影响主请求。
- smart compression 异常自动 `safe_passthrough_repair`。
- rule compressor 异常自动 passthrough。
- prefix reorder 异常自动跳过 reorder。
- provider probe 默认可开启，但探测失败只记录 reason，回退静态 route。
- cheap model 是否真的调用仍受 Profit Guard / Context Guard / Fidelity Guard / Circuit Breaker 约束。
- 最终回答永远由用户主模型生成，cheap model 只压缩上下文。

## 返回 Metadata

响应中会附带 `_optimization`：

- `production_optimizer`
- `rollout_mode`
- `shadow`
- `rollout_gate`
- `l1_smart_compression`
- `prefix_reorder`
- `prefix_hash`
- `cache_stable`
- `cache_savings_estimate`
- `input_tokens_original_est`
- `input_tokens_optimized_est`
- `token_saved_pct_est`
- `final_path`
- `errors`

## 测试覆盖

- `RolloutGate(mode="shadow")` 不启用真实 smart。
- `RolloutGate(mode="auto")` 收益不足拒绝。
- `RolloutGate(mode="auto")` 收益达标通过。
- shadow rollout 发送 rule optimized payload 并返回 metadata。
- `mode="on"` 可触发 smart compressor。
- compressor 异常不破坏主请求，返回 `safe_passthrough_repair`。

## 验证命令

```text
python -m pytest tests/test_production_optimizer.py -q
python -m pytest -q
python benchmark_fidelity_regression.py
python benchmark_l1_v5.py
python benchmark_competitive_adapters.py
python benchmark_shadow_mode.py
```
