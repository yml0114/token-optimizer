# Token Optimizer — Hermes Gateway Plugin

Token 压缩优化插件，为 Hermes Gateway 提供异步非阻塞的对话历史压缩能力。

## 功能

- **异步非阻塞**：用户 <5ms 拿到结果，后台 LLM 静默压缩
- **模型感知压缩**：根据目标模型上下文窗口自动调整压缩比
- **LRU 缓存**：500 条上限，自动淘汰，后续调用 <1ms 命中
- **白名单控制**：只有验证过的模型才触发 LLM 压缩
- **Shadow Mode**：观察模式，只记录不压缩，零风险

## 安装

```bash
# 复制到插件目录
cp __init__.py plugin.yaml ~/.hermes/plugins/token-optimizer/

# 在 config.yaml 中启用
echo "plugins:\n  enabled:\n    - token-optimizer" >> ~/.hermes/config.yaml
```

## 配置

在 `~/.hermes/.env` 中添加：

```bash
TOKEN_OPTIMIZER_ENABLED=1
TOKEN_OPTIMIZER_SHADOW=1              # 1=观察模式，0=正式启用
TOKEN_OPTIMIZER_MIN_INPUT=1000        # 触发压缩的最小 token 数
TOKEN_OPTIMIZER_TARGET_RATIO=0.35     # 默认压缩比
TOKEN_OPTIMIZER_KEEP_RECENT=4         # 保留最近 N 条消息不压缩
TOKEN_OPTIMIZER_LLM_MIN_TOKENS=1500   # 低于此值跳过 LLM 压缩（避免膨胀）
TOKEN_OPTIMIZER_CACHE_MAX=500         # LRU 缓存最大条目数
TOKEN_OPTIMIZER_SMART_MODELS=gpt-5.5,claude-4-sonnet,deepseek-v4-flash,mimo-v2.5
                                      # 白名单：只有这些模型走 LLM 压缩
```

## 模型感知压缩

压缩比根据目标模型上下文窗口自动调整：

| Tier | 上下文窗口 | 压缩比 | 示例模型 |
|------|-----------|--------|---------|
| large | ≥128k | 0.6（轻压） | gpt-5.5, claude-4-sonnet |
| mid | ≥64k | 0.4 | deepseek-v4-flash |
| small | <64k | 0.3（重压） | mimo-v2.5, mimo-v2.5-free |

## 验证

```bash
# 异步+缓存验证
python3 async_cache_test.py

# 模型感知压缩对比
python3 model_aware_test.py
```

## 许可证

MIT
