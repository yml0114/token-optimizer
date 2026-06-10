# Provider Model Probe Report

本轮新增 ProviderModelProbe，用于把 v5 SmartCompressor 从“静态路由表”推进到“可探测同平台 cheap model”的半自动路由。

## 目标

- 调用方显式启用后，安全探测同平台 `/models`。
- 从模型库存里识别 `flash / mini / turbo / lite / haiku / small / compress` 等 cheap compressor 候选。
- 结合本地保守价格表生成 `ModelRoute`。
- 探测失败不阻塞主链路：自动降级到已知静态路由，或记录不可用原因。
- 最终是否调用 cheap model 仍由 Profit Guard / Context Guard / Fidelity Guard 决定。

## 新增文件

- `src/token_optimizer/core/model_probe.py`
- `tests/test_model_probe.py`

## SmartCompressor 集成

`SmartCompressor(..., enable_model_probe=True)` 时：

1. 使用 `ProviderModelProbe(base_url, api_key).probe(main_model)` 探测 `/models`。
2. 若探测得到可用 route，则更新当前 `route / active_option / compressor_model`。
3. metadata 与 shadow telemetry 中输出：
   - `probe.enabled`
   - `probe.available`
   - `probe.source`
   - `probe.provider`
   - `probe.models_seen`
   - `probe.failure_reason`

默认 `enable_model_probe=False`，不产生额外网络请求。

## Safety Behavior

| 场景 | 行为 |
|---|---|
| 未启用 probe | 完全沿用静态 ROUTES |
| `/models` 成功 | 用模型库存发现 cheap candidates |
| `/models` 失败 | 记录 failure_reason，回退静态 route |
| 无 cheap model | `available=False`，SmartCompressor 回退规则路径 |
| 价格未知 | 使用保守 fallback price hint，后续 Profit Guard 决定是否启用 |

## 当前测试覆盖

- provider 推断：MiMo / OpenAI / Anthropic 等。
- OpenAI-like `/models` payload 解析。
- plain list payload 解析。
- MiMo `mimo-v2.5-pro → mimo-v2-flash` 发现。
- 空库存但已知 route 时静态回退。
- 未知模型无 cheap candidate 时返回不可用。
- `/models` 成功 fake client。
- `/models` 失败 fake client 降级静态 route。
- SmartCompressor 启用 probe 后 telemetry 输出 probe metadata。

## 验证结果

```text
python -m pytest tests/test_model_probe.py -q ✅
python -m pytest tests/test_smart_compressor.py tests/test_model_probe.py -q ✅
```
