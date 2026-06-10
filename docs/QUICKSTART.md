# Token Optimizer Production Quickstart

`ProductionOptimizer` 是推荐的生产入口，封装：

- provider model probe
- shadow telemetry
- rollout gate
- smart compression
- rule-only fallback
- protected span / fidelity / profit / context guards
- prefix reorder
- cache metadata
- OpenAI-compatible `/chat/completions` request

## 1. 安装开发版

```bash
pip install -e .
```

## 2. 无外部 API 的本地 smoke test

```bash
python scripts/production_smoke.py
```

预期输出：

```text
production smoke ok
```

## 3. 接入真实 API

```bash
export TOKEN_OPTIMIZER_API_KEY="sk-..."
export TOKEN_OPTIMIZER_BASE_URL="https://api.xiaomimimo.com/v1"
export TOKEN_OPTIMIZER_MODEL="mimo-v2.5-pro"
export TOKEN_OPTIMIZER_ROLLOUT="shadow"
python examples/production_quickstart.py
```

建议上线顺序：

1. `shadow`：只观测，不调用 cheap model。
2. `auto`：收益和风险通过门禁才启用。
3. `on`：强制尝试 smart compression，但仍保留全部 guard。

## 4. 最小代码

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

## 5. Rollout 模式

| mode | 行为 |
|---|---|
| `off` | 不启用 smart compression |
| `shadow` | 只旁路观测 |
| `auto` | 自动门禁，达标才启用 |
| `on` | 强制尝试 smart compression，但 guard 仍生效 |

## 6. 关键安全边界

- cheap model 只压缩，不负责最终回答。
- 最终回答仍由用户主模型生成。
- 任何压缩异常都会 fallback，不破坏主请求。
- 价格未知、上下文不足、收益不足、保真不足都会拒绝 smart compression。
- provider probe 失败不会阻塞请求，只记录 `failure_reason`。
