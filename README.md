<p align="center">
  <strong>Token Optimizer</strong>
</p>

<p align="center">
  <strong>Production-Grade LLM Token Cost Optimization Engine</strong><br>
  Save 70-85% on API costs. Zero quality loss. One-line integration.
</p>

<p align="center">
  <a href="#english">English</a> · <a href="#中文">中文</a> · <a href="https://github.com/yml0114/token-optimizer/tree/main/examples/hermes-plugin">Hermes Plugin</a>
</p>

---

# English

## What is Token Optimizer?

Token Optimizer is a **production-ready token cost optimizer** that sits between your application and any OpenAI-compatible LLM API. It compresses conversation history aggressively while preserving all critical signals — numbers, paths, URLs, emails, code symbols, and constraints.

**Core principle:** The cheap model only compresses. Your main model still does the thinking.

### v2 Highlights (Hermes Plugin)

| Feature | Description |
|---------|-------------|
| **Async Non-Blocking** | User gets rule-only result in <5ms; background LLM compresses silently; next call hits cache in <1ms |
| **Model-Aware Compression** | Compression ratio adapts to target model's context window: large(128k)→0.6, mid(64k)→0.4, small(32k)→0.3 |
| **LRU Cache (500 max)** | Auto-evicts oldest entries; prevents memory bloat on long-running Gateways |
| **SMART_MODELS Whitelist** | Only validated models trigger LLM compression; others get rule-only — zero waste |
| **Inflight Dedup** | Same conversation won't launch duplicate background compressions |
| **Concurrent Tier Racing** | Multiple cheap models race in parallel per tier; first success wins |

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
│  Async Background LLM Compression   │  ← Returns rule-only immediately (<5ms)
│  6,900 → 2,000 tokens               │     Background thread compresses silently
└─────────────────────────────────────┘     Next call hits LRU cache (<1ms)
    │
    ▼
┌─────────────────────────────────────┐
│  Model-Aware Target Ratio           │  ← Adapts compression to target model
│  large(128k)=0.6  mid(64k)=0.4      │     Preserves more context for bigger models
│  small(32k)=0.3                     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Your Main Model                    │  ← Receives 2,000 tokens instead of 10,000
│  (unchanged, same API key)          │     Saves ~80% on input cost
└─────────────────────────────────────┘
```

### Real Benchmark (v2 Async + Cache)

| Metric | Value |
|--------|-------|
| 1st call latency | **2.4ms** (rule-only, user sees result immediately) |
| Background LLM | **25s** (user completely unaware) |
| 2nd call latency | **0.6ms** (cache hit) |
| Token savings | **70-87%** (varies by target model) |
| Model-aware spread | **1.6pp** (86.1% vs 87.7% across 5 targets) |

---

## Quick Start

### Option A: Hermes Plugin (Recommended)

Drop into `~/.hermes/plugins/token-optimizer/` and enable in `config.yaml`:

```yaml
plugins:
  enabled:
    - token-optimizer
```

Environment variables (`.hermes/.env`):

```bash
TOKEN_OPTIMIZER_ENABLED=1
TOKEN_OPTIMIZER_SHADOW=1              # 1=observe only, 0=active
TOKEN_OPTIMIZER_MIN_INPUT=1000        # min tokens to trigger compression
TOKEN_OPTIMIZER_TARGET_RATIO=0.35     # default compression ratio
TOKEN_OPTIMIZER_KEEP_RECENT=4         # recent messages to keep untouched
TOKEN_OPTIMIZER_LLM_MIN_TOKENS=1500   # skip LLM if input below this
TOKEN_OPTIMIZER_CACHE_MAX=500         # LRU cache max entries
TOKEN_OPTIMIZER_SMART_MODELS=gpt-5.5,claude-4-sonnet,deepseek-v4-flash,mimo-v2.5
                                      # whitelist: only these get LLM compression
```

### Option B: Python SDK

```bash
pip install token-optimizer
```

```python
from token_optimizer import ProductionOptimizer, ProductionOptimizerConfig, RolloutGate

optimizer = ProductionOptimizer(ProductionOptimizerConfig(
    model="mimo-v2.5-pro",
    api_key="your-api-key",
    base_url="https://api.xiaomimimo.com/v1",
    rollout=RolloutGate(mode="auto"),
))

response = optimizer.chat_completions_create(
    messages=[{"role": "user", "content": "Analyze this document..."}],
)
print(response["_optimization"]["token_saved_pct_est"])  # e.g. 78.5
```

### Safe Rollout

```
Week 1:  TOKEN_OPTIMIZER_SHADOW=1  → Observe logs, zero risk
Week 2:  TOKEN_OPTIMIZER_SHADOW=0  → Enable after confidence is built
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `TOKEN_OPTIMIZER_ENABLED` | `1` | Master switch |
| `TOKEN_OPTIMIZER_SHADOW` | `0` | Shadow mode: log only, don't compress |
| `TOKEN_OPTIMIZER_MIN_INPUT` | `1000` | Min tokens to trigger compression |
| `TOKEN_OPTIMIZER_TARGET_RATIO` | `0.35` | Default compression ratio |
| `TOKEN_OPTIMIZER_KEEP_RECENT` | `4` | Recent messages to keep untouched |
| `TOKEN_OPTIMIZER_LLM_MIN_TOKENS` | `1500` | Skip LLM if input below this (avoid expansion) |
| `TOKEN_OPTIMIZER_CACHE_MAX` | `500` | LRU cache max entries |
| `TOKEN_OPTIMIZER_SMART_MODELS` | _(empty)_ | Comma-separated whitelist; empty = all models |
| `TOKEN_OPTIMIZER_CHEAP_MODEL` | _(auto)_ | Force specific cheap model |
| `TOKEN_OPTIMIZER_CHEAP_BASE_URL` | _(auto)_ | Force API base URL |
| `TOKEN_OPTIMIZER_CHEAP_API_KEY` | _(auto)_ | Force API key |
| `TOKEN_OPTIMIZER_CONCURRENT_TIMEOUT` | `12` | Per-tier timeout in seconds |
| `TOKEN_OPTIMIZER_CONCURRENT_TIER_SIZE` | `4` | Max candidates per tier race |

---

## 6-Layer Safety Architecture

| # | Safety Layer | What it does | Failure mode |
|---|---|---|---|
| 1 | **Profit-Aware Routing** | Only calls cheap model when predicted savings exceed threshold | Falls back to rule-only (free) |
| 2 | **Short-Input Guard** | Skips cheap model for tiny requests | Returns rule-compressed result |
| 3 | **SMART_MODELS Whitelist** | Only validated models get LLM compression | Others get rule-only |
| 4 | **Semantic Fidelity Guard** | Blocks lossy compression when critical signals dropped | Returns rule-compressed result |
| 5 | **Circuit Breaker** | Stops calling broken cheap models after consecutive failures | Falls back to rule-only, auto-recovers |
| 6 | **Safe Passthrough** | Any uncaught exception → return original messages untouched | Main request proceeds unmodified |

---

## Competitive Positioning

| Feature | Token Optimizer | LLMLingua | Selective Context | Gisting |
|---|---|---|---|---|
| Zero-config auto-routing | ✅ | ❌ | ❌ | ❌ |
| Profit-aware routing | ✅ | ❌ | ❌ | ❌ |
| Async non-blocking | ✅ | ❌ | ❌ | ❌ |
| Model-aware compression | ✅ | ❌ | ❌ | ❌ |
| LRU cache with eviction | ✅ | ❌ | ❌ | ❌ |
| Semantic fidelity guard | ✅ | ❌ | ❌ | ❌ |
| Shadow mode telemetry | ✅ | ❌ | ❌ | ❌ |
| Self-learning circuit breaker | ✅ | ❌ | ❌ | ❌ |
| No model download required | ✅ | ❌ | ✅ | ❌ |
| OpenAI-compatible wrapper | ✅ | ❌ | ❌ | ❌ |
| Cost: rule-only path | Free | — | — | — |
| Cost: smart path | ~$0.0001/req | — | — | — |

---

## License

MIT

---

---

# 中文

## 什么是 Token Optimizer？

Token Optimizer 是一个**生产级 LLM Token 成本优化引擎**，部署在应用与任何 OpenAI 兼容 API 之间。它在保留关键信号（数字、路径、URL、邮箱、代码符号、约束条件）的前提下，激进压缩对话历史。

**核心原则：** 廉价模型只负责压缩，主模型仍然负责思考。

### v2 亮点（Hermes 插件）

| 特性 | 说明 |
|------|------|
| **异步非阻塞** | 用户 <5ms 拿到规则压缩结果，后台 LLM 静默压缩，下次调用 <1ms 命中缓存 |
| **模型感知压缩** | 压缩比自动适配目标模型上下文窗口：大(128k)→0.6，中(64k)→0.4，小(32k)→0.3 |
| **LRU 缓存（500条上限）** | 自动淘汰最旧条目，防止长期运行内存膨胀 |
| **SMART_MODELS 白名单** | 只有验证过的模型才触发 LLM 压缩，其余走规则压缩——零浪费 |
| **防重复压缩** | 同一对话不会启动多个后台压缩线程 |
| **并发 Tier 竞速** | 同一层级多个廉价模型并行竞争，第一个成功的结果胜出 |

### 实测数据（v2 异步+缓存）

| 指标 | 数值 |
|------|------|
| 首次调用延迟 | **2.4ms**（规则压缩，用户立刻拿到结果） |
| 后台 LLM 压缩 | **25s**（用户完全无感） |
| 二次调用延迟 | **0.6ms**（缓存命中） |
| Token 节省 | **70-87%**（因目标模型而异） |
| 模型感知差异 | **1.6 个百分点**（5 个目标模型 86.1%-87.7%） |

### 工作流程

```
用户请求（10,000 tokens）
    │
    ▼
┌─────────────────────────────────────┐
│  规则预压缩（L1 v4）                 │  ← 零成本，去除噪声/填充词
│  10,000 → 6,900 tokens              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  利润感知路由                         │  ← 预测：调用廉价模型划算吗？
│  "mimo-v2-flash 能省 68%？"          │
└─────────────────────────────────────┘
    │  （仅当预测节省 > 阈值时继续）
    ▼
┌─────────────────────────────────────┐
│  异步后台 LLM 压缩                   │  ← 立即返回规则压缩结果（<5ms）
│  6,900 → 2,000 tokens               │     后台线程静默压缩
└─────────────────────────────────────┘     下次调用命中 LRU 缓存（<1ms）
    │
    ▼
┌─────────────────────────────────────┐
│  模型感知目标压缩比                   │  ← 根据目标模型自动调整压缩力度
│  大上下文(128k)=0.6  中(64k)=0.4     │     大模型保留更多细节
│  小上下文(32k)=0.3                   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  主模型                              │  ← 收到 2,000 tokens 而非 10,000
│  （不变，同一个 API key）             │     输入成本节省 ~80%
└─────────────────────────────────────┘
```

### 白名单机制

`TOKEN_OPTIMIZER_SMART_MODELS` 控制哪些模型可以触发 LLM 智能压缩：

```bash
# 只有这些模型走 LLM 压缩，其余一律 rule-only
TOKEN_OPTIMIZER_SMART_MODELS=gpt-5.5,claude-4-sonnet,deepseek-v4-flash,mimo-v2.5,mimo-v2.5-free

# 留空 = 所有模型都走智能压缩（向后兼容）
TOKEN_OPTIMIZER_SMART_MODELS=
```

**为什么用白名单？**
- 不同模型对压缩上下文的容忍度不同，未验证的模型压缩后质量可能断崖下降
- 避免浪费 API 调用在不需要压缩的模型上
- 安全可控：只对你验证过的模型开启

### 安全架构

| # | 安全层 | 作用 | 失败回退 |
|---|--------|------|----------|
| 1 | **利润感知路由** | 只在预测节省超过阈值时调用廉价模型 | 回退到规则压缩（免费） |
| 2 | **短输入保护** | 跳过过小的请求 | 返回规则压缩结果 |
| 3 | **SMART_MODELS 白名单** | 只有验证过的模型才走 LLM 压缩 | 其余走规则压缩 |
| 4 | **语义保真守卫** | 关键信号被丢弃时阻止有损压缩 | 返回规则压缩结果 |
| 5 | **熔断器** | 连续失败后停止调用损坏的廉价模型 | 回退到规则压缩，自动恢复 |
| 6 | **安全透传** | 任何未捕获异常 → 返回原始消息 | 主请求不受影响 |

---

## 许可证

MIT

## Public API smoke coverage

v2.16 also introduces a thin public API layer for library integration:

```python
from token_optimizer import compress_text

result = compress_text(
    "Owner Liang. Deadline Jun 25. Budget is 900 USD.",
    mode="safe",
    content_type="memory",
    preserve=["numbers", "dates", "identifiers"],
)

print(result.compressed)
print(result.stats)
print(result.quality.miss_summary)
```

`compress_text()` is intentionally scoped to caller-selected text. It does not retrieve memory, rank context blocks, assemble prompts, or manage long-term context.
