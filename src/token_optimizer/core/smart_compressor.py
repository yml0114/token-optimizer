"""L1 v5: SmartCompressor — Profit-Aware same-key smart compression.

前置条件（全部满足才激活模型压缩，否则自动退化为 v4 纯规则）：
  1. main_model 能映射到同平台廉价模型候选（见 ROUTES）
  2. api_key 非空（用同一个 key 调廉价模型）
  3. base_url 非空（同一平台，同一 API endpoint）
  4. 真实/准真实 token 估算后，预计成本收益为正且超过 min_profit_margin
  5. API 调用成功、输出通过校验、实际成本仍然收益为正

若任一条件不满足 → 纯规则压缩（v4，零成本）

核心原则：
  廉价模型只负责压缩，最终解决问题的一直是用户的主模型。
  没有廉价模型、廉价模型不可用、或者算账不划算，都不会硬凑。

MiMo 特别说明：
  MiMo V2.5 / V2.5 Pro 没有同代 Flash。
  这里使用的是同平台仍可调用的跨代廉价模型 mimo-v2-flash。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx

from token_optimizer.core.signal_noise import (
    CompressionLevel,
    InputCompressor,
)


# ══════════════════════════════════════════════════════════════════════════════
# Pricing / Routing
# ══════════════════════════════════════════════════════════════════════════════

TOKENS_PER_MILLION = 1_000_000
DEFAULT_OUTPUT_RATIO = 0.20
DEFAULT_SAFE_SMART_TARGET_RATIO = 0.30
DEFAULT_EXTREME_SMART_TARGET_RATIO = 0.22
DEFAULT_PROTECTED_SMART_TARGET_RATIO = 0.45
DEFAULT_SMART_TARGET_RATIO = DEFAULT_SAFE_SMART_TARGET_RATIO
DEFAULT_MIN_PROFIT_MARGIN = 0.05  # require at least 5% cheaper than rule-only
DEFAULT_MIN_FIDELITY_SCORE = 0.78
DEFAULT_PROTECTED_MIN_FIDELITY_SCORE = 0.88


@dataclass(frozen=True)
class CheapModelOption:
    """A cheap compressor candidate on the same API platform."""

    model: str
    input_price: float
    output_price: float
    max_context: int = 1_000_000
    cross_generation: bool = False
    note: str = ""


@dataclass(frozen=True)
class ModelRoute:
    """A same-platform main-model route with one or more cheap candidates.

    All prices are USD per 1M tokens. Cheap candidates share the same API key and
    base_url with the main model. The router chooses the best profitable candidate
    at runtime instead of blindly using the first match.
    """

    pattern: str
    main_input_price: float
    main_output_price: float
    cheap_options: tuple[CheapModelOption, ...]

    @property
    def cheap_model(self) -> str:
        """Backward-compatible primary cheap model name."""
        return self.cheap_options[0].model

    @property
    def cheap_input_price(self) -> float:
        return self.cheap_options[0].input_price

    @property
    def cheap_output_price(self) -> float:
        return self.cheap_options[0].output_price

    @property
    def cheap_max_context(self) -> int:
        return self.cheap_options[0].max_context

    @property
    def cross_generation(self) -> bool:
        return self.cheap_options[0].cross_generation

    @property
    def note(self) -> str:
        return self.cheap_options[0].note


@dataclass
class CandidateLearningStats:
    """Runtime feedback for a cheap compressor candidate.

    This is deliberately local/in-memory: it makes the current process safer without
    needing a database. Callers can later persist this if they want fleet learning.
    """

    attempts: int = 0
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    total_ratio: float = 0.0
    total_savings_pct: float = 0.0
    disabled_until_call: int = 0
    last_reason: str = ""

    @property
    def avg_ratio(self) -> float | None:
        return self.total_ratio / self.successes if self.successes else None

    @property
    def avg_savings_pct(self) -> float | None:
        return self.total_savings_pct / self.successes if self.successes else None


@dataclass(frozen=True)
class CompressionPolicy:
    """Risk-aware compression target.

    safe: default production target.
    extreme: low-risk noisy/history/tool content can be compressed harder.
    protected: high-risk code/errors/numbers/legal/finance/medical content keeps more context.
    """

    mode: str
    target_ratio: float
    reason: str
    risk_score: int


@dataclass(frozen=True)
class FidelityReport:
    """Cheap deterministic semantic-fidelity guard report."""

    score: float
    passed: bool
    threshold: float
    coverage: dict[str, float]
    missing: dict[str, list[str]]
    original_signal_count: dict[str, int]


@dataclass(frozen=True)
class ProtectedSpan:
    """A concrete text span that must survive smart compression."""

    kind: str
    value: str
    occurrences: int = 1


HIGH_RISK_KEYWORDS = (
    "```", "traceback", "exception", "error", "api_key", "token", "password",
    "http://", "https://", "file:", "/app/", "/users/", "def ", "class ",
    "import ", "select ", "insert ", "update ", "delete ", "金融", "股票", "投资",
    "法律", "合同", "医疗", "诊断", "药", "价格", "成本", "$", "¥", "%",
)

LOW_RISK_KEYWORDS = (
    "谢谢", "你好", "好的", "不客气", "顺便", "就是说", "其实", "反正",
    "当然可以", "希望", "大概", "简单说", "总结", "历史", "背景", "metadata",
    "trace_id", "log_id", "status", "encoding", "file_size",
)


def assess_compression_policy(messages: list[dict[str, Any]]) -> CompressionPolicy:
    """Choose safe/extreme/protected compression target from message content.

    This is a cheap pre-flight gate. It does not try to understand everything; it
    only prevents obviously risky content from being over-compressed while allowing
    repetitive/noisy context to use a more aggressive target.
    """
    text_parts: list[str] = []
    tool_chars = 0
    assistant_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        text_parts.append(content.lower())
        if msg.get("role") == "tool":
            tool_chars += len(content)
        if msg.get("role") == "assistant":
            assistant_chars += len(content)

    joined = "\n".join(text_parts)
    risk_score = sum(1 for kw in HIGH_RISK_KEYWORDS if kw.lower() in joined)
    noise_score = sum(1 for kw in LOW_RISK_KEYWORDS if kw.lower() in joined)
    total_chars = max(1, sum(len(x) for x in text_parts))
    tool_or_assistant_ratio = (tool_chars + assistant_chars) / total_chars

    if risk_score >= 2:
        return CompressionPolicy(
            mode="protected",
            target_ratio=DEFAULT_PROTECTED_SMART_TARGET_RATIO,
            reason="检测到代码/错误/数字/路径/金融法律医疗等高风险信息，启用保护阀",
            risk_score=risk_score,
        )
    if noise_score >= 3 or tool_or_assistant_ratio > 0.55:
        return CompressionPolicy(
            mode="extreme",
            target_ratio=DEFAULT_EXTREME_SMART_TARGET_RATIO,
            reason="检测到高冗余历史/工具噪声/寒暄解释，启用激进阀",
            risk_score=risk_score,
        )
    return CompressionPolicy(
        mode="safe",
        target_ratio=DEFAULT_SAFE_SMART_TARGET_RATIO,
        reason="默认生产安全阀",
        risk_score=risk_score,
    )


def _mimo_flash(note: str) -> CheapModelOption:
    return CheapModelOption(
        model="mimo-v2-flash",
        input_price=0.10,
        output_price=0.30,
        max_context=256_000,
        cross_generation=True,
        note=note,
    )


ROUTES: tuple[ModelRoute, ...] = (
    # MiMo V2.5 has no same-generation Flash. Use still-available V2 Flash.
    ModelRoute(
        pattern="mimo-v2.5-pro",
        main_input_price=1.00,
        main_output_price=3.00,
        cheap_options=(
            _mimo_flash("MiMo V2.5 Pro → V2 Flash：跨代，但同平台同 key，Flash 仍可调用"),
        ),
    ),
    ModelRoute(
        pattern="mimo-v2.5",
        main_input_price=0.14,
        main_output_price=0.28,
        cheap_options=(
            _mimo_flash("MiMo V2.5 → V2 Flash：主模型本身已很便宜，必须先算账，收益不足则回退 v4"),
        ),
    ),
    ModelRoute(
        pattern="mimo-pro",
        main_input_price=1.00,
        main_output_price=3.00,
        cheap_options=(
            _mimo_flash("MiMo Pro alias → V2 Flash"),
        ),
    ),

    # DeepSeek / Qwen / OpenAI / Anthropic examples. Prices are conservative defaults;
    # profit gate protects users if a route becomes uneconomical.
    ModelRoute(
        "deepseek-v4-pro",
        0.435,
        0.87,
        (CheapModelOption("deepseek-v4-flash", 0.14, 0.28),),
    ),
    ModelRoute(
        "deepseek-pro",
        0.435,
        0.87,
        (CheapModelOption("deepseek-v4-flash", 0.14, 0.28),),
    ),
    ModelRoute(
        "qwen-max",
        2.40,
        9.60,
        (CheapModelOption("qwen-turbo", 0.05, 0.20),),
    ),
    ModelRoute(
        "qwen-plus",
        0.40,
        1.20,
        (CheapModelOption("qwen-turbo", 0.05, 0.20),),
    ),
    ModelRoute(
        "gpt-4o",
        2.50,
        10.00,
        (CheapModelOption("gpt-4o-mini", 0.15, 0.60),),
    ),
    ModelRoute(
        "gpt-4-turbo",
        10.00,
        30.00,
        (CheapModelOption("gpt-4o-mini", 0.15, 0.60),),
    ),
    ModelRoute(
        "claude-3-opus",
        15.00,
        75.00,
        (CheapModelOption("claude-3-haiku", 0.25, 1.25),),
    ),
    ModelRoute(
        "claude-3.5-sonnet",
        3.00,
        15.00,
        (CheapModelOption("claude-3-haiku", 0.25, 1.25),),
    ),
)


def find_route(model: str) -> ModelRoute | None:
    """Find a same-platform cheap-model route for the user's main model."""
    model_lower = model.lower().strip()
    for route in ROUTES:
        if route.pattern in model_lower:
            return route
    return None


def find_cheap_sibling(model: str) -> str | None:
    """Backward-compatible helper: return primary cheap model name or None."""
    route = find_route(model)
    return route.cheap_model if route else None


# ══════════════════════════════════════════════════════════════════════════════
# Token estimation
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=32)
def _get_tiktoken_encoding(model: str):  # pragma: no cover - depends on optional package
    """Return a tiktoken encoding when installed, otherwise None.

    tiktoken is optional: production users who install it get closer token counts;
    lightweight installs keep deterministic fallback behavior.
    """
    try:
        import tiktoken  # type: ignore
    except Exception:
        return None

    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def estimate_tokens_from_text(text: str, model: str = "") -> int:
    """Estimate tokens with real tokenizer when available, fallback otherwise.

    Fallback is multilingual-aware instead of the old raw ``len(text)//3``:
    - CJK characters are denser and often close to 1 char/token.
    - ASCII words are closer to ~4 chars/token.
    - Other unicode sits in between.
    """
    if not text:
        return 0

    encoding = _get_tiktoken_encoding(model) if model else None
    if encoding is not None:  # pragma: no cover - optional dependency path
        return max(1, len(encoding.encode(text)))

    ascii_chars = 0
    cjk_chars = 0
    other_chars = 0
    for ch in text:
        code = ord(ch)
        if code < 128:
            ascii_chars += 1
        elif 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            cjk_chars += 1
        else:
            other_chars += 1

    tokens = int(ascii_chars / 4.0 + cjk_chars / 1.7 + other_chars / 2.5)
    return max(1, tokens)


def estimate_tokens_from_messages(
    messages: list[dict[str, Any]],
    model: str = "",
    include_message_overhead: bool = True,
) -> int:
    """Estimate chat tokens from messages.

    Includes lightweight chat-message overhead so cost guards don't undercount many
    short messages. Uses tiktoken if installed; otherwise multilingual fallback.
    """
    total = 0
    for msg in messages:
        if include_message_overhead:
            total += 4  # role/name/separators; conservative chat envelope estimate
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(role, str):
            total += estimate_tokens_from_text(role, model=model)
        if isinstance(content, str):
            total += estimate_tokens_from_text(content, model=model)
    return max(0, total)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_m: float,
    output_price_per_m: float,
) -> float:
    """Estimate model call cost in USD."""
    return (
        input_tokens / TOKENS_PER_MILLION * input_price_per_m
        + output_tokens / TOKENS_PER_MILLION * output_price_per_m
    )


def estimate_route_profit(
    route: ModelRoute,
    rule_tokens: int,
    smart_tokens: int,
    output_ratio: float = DEFAULT_OUTPUT_RATIO,
    cheap_option: CheapModelOption | None = None,
) -> dict[str, float | bool | str]:
    """Compare rule-only cost vs cheap-compressor + main-model cost.

    Rule-only path:
        main_model(rule_tokens → answer)

    Smart path:
        cheap_model(rule_tokens → smart_tokens) + main_model(smart_tokens → answer)
    """
    option = cheap_option or route.cheap_options[0]
    rule_output_tokens = int(rule_tokens * output_ratio)
    smart_answer_tokens = int(smart_tokens * output_ratio)

    rule_cost = estimate_cost(
        rule_tokens,
        rule_output_tokens,
        route.main_input_price,
        route.main_output_price,
    )
    compressor_cost = estimate_cost(
        rule_tokens,
        smart_tokens,
        option.input_price,
        option.output_price,
    )
    main_after_cost = estimate_cost(
        smart_tokens,
        smart_answer_tokens,
        route.main_input_price,
        route.main_output_price,
    )
    smart_total_cost = compressor_cost + main_after_cost
    savings = rule_cost - smart_total_cost
    savings_pct = (savings / rule_cost * 100) if rule_cost > 0 else 0.0

    return {
        "candidate": option.model,
        "rule_cost": round(rule_cost, 8),
        "smart_total_cost": round(smart_total_cost, 8),
        "compressor_cost": round(compressor_cost, 8),
        "main_after_cost": round(main_after_cost, 8),
        "savings": round(savings, 8),
        "savings_pct": round(savings_pct, 2),
        "profitable": savings > 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Fidelity guards
# ══════════════════════════════════════════════════════════════════════════════

_SIGNAL_PATTERNS: dict[str, str] = {
    "numbers": r"(?<![\w.])(?:\d+(?:\.\d+)?%?|[$¥]\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:ms|s|kg|mb|gb|k|m|万|亿))(?![\w.])",
    "paths": r"(?:/[\w.\-\u4e00-\u9fff]+){2,}|[A-Za-z]:\\(?:[^\\\s]+\\?)+|[\w.\-]+\.(?:py|js|ts|tsx|jsx|json|md|yaml|yml|txt|csv|xlsx|pdf)",
    "urls": r"https?://[^\s)\]}>\"，。；：、]+",
    "emails": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "code_symbols": r"\b(?:def|class|import|from|return|async|await|function|const|let|var|SELECT|INSERT|UPDATE|DELETE)\b|[A-Za-z_][A-Za-z0-9_]{2,}\(\)?",
    "constraints": r"(?:必须|不能|不要|禁止|一定|只允许|至少|最多|保留|完整|优先|不得|must|never|only|required|forbidden)",
}

_SIGNAL_WEIGHTS: dict[str, float] = {
    "numbers": 0.20,
    "paths": 0.20,
    "urls": 0.12,
    "emails": 0.12,
    "code_symbols": 0.20,
    "constraints": 0.18,
    "latest_user_keywords": 0.10,
}

_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "请", "帮我", "一个", "这个", "那个", "就是", "可以", "需要", "进行", "我们", "你", "我",
    "谢谢", "你好", "好的", "就是说", "大概", "一下", "请只", "不需要", "解释", "保留", "核心", "代码",
}


def _messages_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(msg.get("content", "")) for msg in messages if isinstance(msg.get("content", ""), str)
    )


def _unique_matches(pattern: str, text: str, *, flags: int = re.IGNORECASE) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for match in re.findall(pattern, text, flags):
        value = match if isinstance(match, str) else "".join(match)
        value = value.strip().strip('.,;:，。；：')
        if not value:
            continue
        key = value.lower()
        if key not in seen:
            seen.add(key)
            values.append(value)
    return values


def _latest_user_keywords(messages: list[dict[str, Any]], limit: int = 12) -> list[str]:
    latest = ""
    for msg in reversed(messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            latest = msg["content"]
            break
    if not latest:
        return []

    keywords: list[str] = []
    seen: set[str] = set()

    # ASCII identifiers / product names / function-like terms.
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{3,}", latest):
        key = token.lower()
        if key not in _STOPWORDS and key not in seen:
            seen.add(key)
            keywords.append(token)

    # CJK: first capture common semantic compounds; then fall back to compact chunks.
    cjk_runs = re.findall(r"[\u4e00-\u9fff]{2,}", latest)
    semantic_terms = (
        "项目上下文", "上下文", "长内容", "核心代码", "快速排序", "排序函数",
        "历史背景", "错误信息", "测试用例", "边界条件", "压缩", "总结", "代码",
    )
    for term in semantic_terms:
        if term in latest and term not in seen:
            seen.add(term)
            keywords.append(term)
            if len(keywords) >= limit:
                return keywords

    for run in cjk_runs:
        # Repeated test prompts often concatenate the same sentence many times.
        # Limit the window to avoid extracting boundary artifacts like "内容请帮".
        compact = run[: min(len(run), 24)]
        for size in (4, 3, 2):
            for i in range(0, max(1, len(compact) - size + 1), size):
                token = compact[i:i + size]
                if len(token) < 2:
                    continue
                key = token.lower()
                if key in _STOPWORDS or key in seen:
                    continue
                if any(stop in key for stop in ("谢谢", "你好", "就是", "大概", "请帮", "请压", "不需要解释")):
                    continue
                seen.add(key)
                keywords.append(token)
                if len(keywords) >= limit:
                    return keywords
    return keywords[:limit]


def extract_critical_signals(messages: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Extract deterministic critical signals that compression must preserve."""
    text = _messages_text(messages)
    signals = {name: _unique_matches(pattern, text)[:32] for name, pattern in _SIGNAL_PATTERNS.items()}
    # Single generic modal words like "保留" / "必须" are too broad to require
    # verbatim preservation. Concrete objects around them are covered by latest
    # user keywords and hard signals.
    generic_constraints = {"必须", "不能", "不要", "禁止", "一定", "保留", "完整", "优先", "不得"}
    signals["constraints"] = [
        value for value in signals.get("constraints", [])
        if value not in generic_constraints and len(value) > 2
    ]
    signals["latest_user_keywords"] = _latest_user_keywords(messages)
    return signals


def extract_protected_spans(messages: list[dict[str, Any]], limit: int = 64) -> list[ProtectedSpan]:
    """Return concrete spans that a model compressor must preserve verbatim.

    These spans are deterministic and zero-cost. They are stronger than generic
    keywords: numbers, paths, URLs, emails, code symbols, and non-generic hard
    constraints should be copied into the compressed prompt whenever present.
    """
    signals = extract_critical_signals(messages)
    ordered_kinds = ("urls", "emails", "paths", "code_symbols", "numbers", "constraints")
    spans: list[ProtectedSpan] = []
    seen: set[tuple[str, str]] = set()
    full_text = _messages_text(messages).lower()
    for kind in ordered_kinds:
        for value in signals.get(kind, []):
            normalized = value.strip()
            if not normalized:
                continue
            key = (kind, normalized.lower())
            if key in seen:
                continue
            seen.add(key)
            occurrences = max(1, full_text.count(normalized.lower()))
            spans.append(ProtectedSpan(kind=kind, value=normalized, occurrences=occurrences))
            if len(spans) >= limit:
                return spans
    return spans


def format_protected_spans(spans: list[ProtectedSpan], limit: int = 64) -> str:
    """Format protected spans for the cheap model prompt."""
    if not spans:
        return ""
    lines = ["PROTECTED_SPANS: 以下字段必须原样保留，不能改写、合并、删除："]
    for span in spans[:limit]:
        lines.append(f"- [{span.kind}] {span.value}")
    return "\n".join(lines)


def score_semantic_fidelity(
    original: list[dict[str, Any]],
    compressed: list[dict[str, Any]],
    *,
    policy: CompressionPolicy | None = None,
) -> FidelityReport:
    """Score whether compressed messages preserve high-value task signals.

    This is not a replacement for LLM judge; it is a zero-cost production guard.
    If it fails, we do not send the lossy compressed result to the main model.
    """
    original_signals = extract_critical_signals(original)
    compressed_text = _messages_text(compressed).lower()
    coverage: dict[str, float] = {}
    missing: dict[str, list[str]] = {}

    weighted_score = 0.0
    active_weight = 0.0
    for name, values in original_signals.items():
        weight = _SIGNAL_WEIGHTS[name]
        if not values:
            coverage[name] = 1.0
            continue
        if name == "latest_user_keywords":
            kept = [value for value in values if value.lower() in compressed_text]
            # For low-risk natural-language asks, preserving the main intent keywords is enough;
            # do not require every extracted filler-adjacent chunk to survive verbatim.
            coverage[name] = min(1.0, len(kept) / max(1, min(3, len(values))))
            missed = [value for value in values if value.lower() not in compressed_text]
        else:
            kept = [value for value in values if value.lower() in compressed_text]
            missed = [value for value in values if value.lower() not in compressed_text]
            coverage[name] = len(kept) / len(values)
        if missed:
            missing[name] = missed[:8]
        weighted_score += coverage[name] * weight
        active_weight += weight

    score = weighted_score / active_weight if active_weight > 0 else 1.0
    hard_signal_count = sum(
        len(original_signals.get(name, []))
        for name in ("numbers", "paths", "urls", "emails", "code_symbols")
    )
    if policy and policy.mode == "protected":
        threshold = DEFAULT_PROTECTED_MIN_FIDELITY_SCORE
    elif hard_signal_count == 0:
        # Natural-language summaries should not be blocked just because a few
        # low-value phrase chunks were paraphrased. Hard signals still use the
        # normal threshold.
        threshold = min(DEFAULT_MIN_FIDELITY_SCORE, 0.55)
    else:
        threshold = DEFAULT_MIN_FIDELITY_SCORE
    return FidelityReport(
        score=round(score, 4),
        passed=score >= threshold,
        threshold=threshold,
        coverage={k: round(v, 4) for k, v in coverage.items()},
        missing=missing,
        original_signal_count={k: len(v) for k, v in original_signals.items()},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Compression prompt
# ══════════════════════════════════════════════════════════════════════════════

COMPRESSION_SYSTEM_PROMPT = """你是一个精确的输入压缩器。压缩对话消息列表，最少token保留所有关键信息。

目标：
- 默认把输入压到规则压缩结果的 30% 左右。
- 高冗余历史、礼貌语、重复解释、工具噪声可压到 20%-25%。
- 代码、错误栈、API 参数、文件路径、数字、约束条件不足以安全压缩时，宁可放宽到 40%-50%。

绝对边界：
1. 你只压缩，不解决用户问题
2. 不新增事实，不改写任务目标，不改变代码语义
3. system 消息必须完整保留
4. 最近 1 轮用户目标完整保留；旧轮只保留核心指令、关键结论、重要代码片段
5. 错误信息只保留错误类型、文件、行号、关键堆栈
6. 工具输出只保留可复现任务所需的关键数据
7. 多轮历史要合并重复意图，保留最终决定，不保留寒暄和过程性解释
8. 如果压缩会导致信息丢失，宁可少压缩
9. 数字、路径、URL、API参数、函数名、错误类型、强约束词必须尽量原样保留

输出要求：
- 保持 JSON 数组格式，元素 {"role": "...", "content": "..."}
- 能合并的旧 assistant/user 消息可合并为一条摘要，但必须保留至少一条 user 消息
- 仅输出压缩后的JSON数组，不要解释，不要 markdown。"""


# ══════════════════════════════════════════════════════════════════════════════
# SmartCompressor
# ══════════════════════════════════════════════════════════════════════════════

class SmartCompressor:
    """Profit-aware smart compressor.

    Cheap model only compresses input. The user's main model still performs the
    actual reasoning / answer generation.
    """

    def __init__(
        self,
        main_model: str = "",
        api_key: str = "",
        base_url: str = "",
        timeout: float = 30.0,
        level: CompressionLevel = CompressionLevel.AGGRESSIVE,
        min_profit_margin: float = DEFAULT_MIN_PROFIT_MARGIN,
        min_rule_tokens_for_smart: int = 128,
        expected_smart_ratio: float = DEFAULT_SMART_TARGET_RATIO,
        safe_target_ratio: float = DEFAULT_SAFE_SMART_TARGET_RATIO,
        extreme_target_ratio: float = DEFAULT_EXTREME_SMART_TARGET_RATIO,
        protected_target_ratio: float = DEFAULT_PROTECTED_SMART_TARGET_RATIO,
        learning_enabled: bool = True,
        max_consecutive_failures: int = 2,
        circuit_breaker_cooldown: int = 20,
        min_fidelity_score: float = DEFAULT_MIN_FIDELITY_SCORE,
        protected_min_fidelity_score: float = DEFAULT_PROTECTED_MIN_FIDELITY_SCORE,
        enable_model_probe: bool = False,
    ):
        self.main_model = main_model
        self.level = level
        self.rule_compressor = InputCompressor(level=level)
        self.timeout = timeout
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.route = find_route(main_model) if main_model else None
        self.active_option: CheapModelOption | None = self.route.cheap_options[0] if self.route else None
        self.compressor_model = self.active_option.model if self.active_option else ""
        self.min_profit_margin = min_profit_margin
        self.min_rule_tokens_for_smart = min_rule_tokens_for_smart
        self.expected_smart_ratio = expected_smart_ratio
        self.safe_target_ratio = safe_target_ratio
        self.extreme_target_ratio = extreme_target_ratio
        self.protected_target_ratio = protected_target_ratio
        self.current_policy: CompressionPolicy | None = None
        self.learning_enabled = learning_enabled
        self.max_consecutive_failures = max_consecutive_failures
        self.circuit_breaker_cooldown = circuit_breaker_cooldown
        self.learning_stats: dict[str, CandidateLearningStats] = {}
        self.min_fidelity_score = min_fidelity_score
        self.protected_min_fidelity_score = protected_min_fidelity_score
        self.current_protected_spans: list[ProtectedSpan] = []
        self._compress_calls = 0

        self.enable_model_probe = enable_model_probe
        self.probe_result = None
        if enable_model_probe and main_model:
            from token_optimizer.core.model_probe import ProviderModelProbe
            self.probe_result = ProviderModelProbe(base_url, api_key, timeout=min(timeout, 10.0)).probe(main_model)
            if self.probe_result.route is not None:
                self.route = self.probe_result.route
                self.active_option = self.route.cheap_options[0]
                self.compressor_model = self.active_option.model

        self.is_configured = bool(self.route and api_key and base_url)

    def compress(
        self,
        messages: list[dict[str, Any]],
        system_text: str = "",
    ) -> tuple[list[dict[str, Any]], dict]:
        """Compress messages using rule-only or profit-aware smart compression."""
        self._compress_calls += 1
        try:
            rule_result, rule_meta = self.rule_compressor.compress_messages(
                messages, system_text=system_text
            )
        except Exception as e:
            # Self-repair guard: compression must never break the main request path
            # or cause extra model spend. Return original messages and avoid API calls.
            return messages, self._meta(
                mode="safe_passthrough_repair",
                reason=f"规则压缩器异常，已安全旁路且未调用廉价模型: {str(e)[:200]}",
                rule_meta={"error": str(e)[:200]},
            )

        self.current_policy = self._assess_policy(rule_result)
        self.current_protected_spans = extract_protected_spans(messages)

        if not self.is_configured or not self.route:
            return rule_result, self._meta(
                mode="rule_only",
                reason="无同平台廉价模型可用或缺少API配置",
                rule_meta=rule_meta,
            )

        rule_tokens = rule_meta.get("compressed_tokens_est") or estimate_tokens_from_messages(
            rule_result, model=self.main_model
        )
        if rule_tokens < self.min_rule_tokens_for_smart:
            return rule_result, self._meta(
                mode="rule_only_profit_guard",
                reason="规则压缩后输入过短，调用廉价模型的固定成本不划算",
                rule_meta=rule_meta,
                projected=self._project_profit(rule_tokens),
            )

        try:
            selected, projected, diagnostics = self._select_best_option(rule_tokens)
        except Exception as e:
            if selected is not None:
                self._record_learning(selected, success=False, reason="api_exception")
            return rule_result, self._meta(
                mode="rule_only_self_repair",
                reason=f"候选路由器异常，已安全回退纯规则且未调用廉价模型: {str(e)[:200]}",
                rule_meta=rule_meta,
            )

        if selected is None:
            all_context_blocked = bool(diagnostics) and all(
                item.get("blocked_by_context") for item in diagnostics
            )
            all_circuit_blocked = bool(diagnostics) and all(
                item.get("blocked_by_circuit") for item in diagnostics
            )
            return rule_result, self._meta(
                mode=(
                    "rule_only_context_guard" if all_context_blocked
                    else "rule_only_self_repair" if all_circuit_blocked
                    else "rule_only_profit_guard"
                ),
                reason=(
                    "规则压缩后仍超过所有廉价模型上下文窗口，回退主模型纯规则路径"
                    if all_context_blocked else
                    "廉价模型连续失败，熔断自修复中，本次回退纯规则以避免费用放大"
                    if all_circuit_blocked else
                    "所有廉价模型候选预测收益不足，不调用模型压缩"
                ),
                rule_meta=rule_meta,
                projected=projected,
                candidate_diagnostics=diagnostics,
            )

        self.active_option = selected
        self.compressor_model = selected.model

        try:
            smart_result = self._call_compressor(rule_result, policy=self.current_policy)

            if not self._validate(smart_result, messages):
                self._record_learning(selected, success=False, reason="validation_failed")
                return rule_result, self._meta(
                    mode="rule_only_fallback",
                    reason="廉价模型输出校验未通过（过长/丢失system/缺失user）",
                    rule_meta=rule_meta,
                    projected=projected,
                    candidate_diagnostics=diagnostics,
                )

            fidelity = self._score_fidelity(messages, smart_result)
            if not fidelity.passed:
                self._record_learning(selected, success=False, reason="fidelity_guard_failed")
                return rule_result, self._meta(
                    mode="rule_only_fidelity_guard",
                    reason="语义保真评分未通过，回退纯规则以避免关键信息丢失",
                    rule_meta=rule_meta,
                    projected=projected,
                    candidate_diagnostics=diagnostics,
                    fidelity_report=fidelity,
                )

            final_tokens = estimate_tokens_from_messages(smart_result, model=self.main_model)
            actual_profit = estimate_route_profit(
                self.route,
                rule_tokens,
                final_tokens,
                cheap_option=selected,
            )
            if (
                not actual_profit["profitable"]
                or actual_profit["savings_pct"] < self.min_profit_margin * 100
            ):
                self._record_learning(
                    selected,
                    success=False,
                    reason="actual_profit_insufficient",
                    rule_tokens=rule_tokens,
                    final_tokens=final_tokens,
                    actual_profit=actual_profit,
                )
                return rule_result, self._meta(
                    mode="rule_only_profit_guard",
                    reason="实际压缩结果收益不足，回退纯规则",
                    rule_meta=rule_meta,
                    projected=projected,
                    actual_profit=actual_profit,
                    candidate_diagnostics=diagnostics,
                )

            self._record_learning(
                selected,
                success=True,
                reason="success",
                rule_tokens=rule_tokens,
                final_tokens=final_tokens,
                actual_profit=actual_profit,
            )

            return smart_result, self._meta(
                mode="smart",
                reason="收益校验通过：廉价模型只压缩，主模型负责最终推理",
                rule_meta=rule_meta,
                projected=projected,
                actual_profit=actual_profit,
                candidate_diagnostics=diagnostics,
                fidelity_report=fidelity,
                smart_compression={
                    "model": self.compressor_model,
                    "input_tokens": rule_tokens,
                    "output_tokens": final_tokens,
                    "final_tokens": final_tokens,
                    "original_tokens": rule_meta.get("original_tokens_est", 0),
                    "total_savings_pct": actual_profit["savings_pct"],
                },
            )

        except Exception as e:
            if selected is not None:
                self._record_learning(selected, success=False, reason="api_exception")
            return rule_result, self._meta(
                mode="rule_only_fallback",
                reason=f"廉价模型调用失败: {str(e)[:200]}",
                rule_meta=rule_meta,
                projected=projected,
                candidate_diagnostics=diagnostics,
            )

    def shadow_evaluate(
        self,
        messages: list[dict[str, Any]],
        system_text: str = "",
    ) -> dict:
        """Dry-run v5 routing without calling the cheap compressor model.

        Shadow mode is meant for production telemetry before rollout: it keeps the
        real request path unchanged and records what v5 *would* do, including
        projected savings, policy risk, protected spans, selected candidate, and
        fallback reason. It must never call ``_call_compressor`` or mutate learning
        feedback.
        """
        try:
            rule_result, rule_meta = self.rule_compressor.compress_messages(
                messages, system_text=system_text
            )
        except Exception as e:
            original_tokens = estimate_tokens_from_messages(messages, model=self.main_model)
            return {
                "mode": "shadow",
                "enabled": True,
                "would_call_smart": False,
                "would_use_rule_only": False,
                "would_fallback_reason": "rule_compressor_exception",
                "reason": f"规则压缩器异常，真实请求应安全旁路且不调用廉价模型: {str(e)[:200]}",
                "original_tokens": original_tokens,
                "rule_tokens": original_tokens,
                "estimated_smart_tokens": None,
                "estimated_raw_cost": None,
                "estimated_rule_cost": None,
                "estimated_smart_cost": None,
                "estimated_savings_pct": None,
                "policy_mode": None,
                "policy_target_ratio": None,
                "protected_span_count": 0,
                "protected_span_kinds": [],
                "route": None,
                "selected_candidate": None,
                "candidate_diagnostics": [],
                "risk_flags": ["safe_passthrough_repair"],
                "rule_compression": {"error": str(e)[:200]},
                "notes": "shadow mode does not modify the request and does not call cheap models",
            }

        policy = self._assess_policy(rule_result)
        spans = extract_protected_spans(messages)
        original_tokens = rule_meta.get("original_tokens_est") or estimate_tokens_from_messages(
            messages, model=self.main_model
        )
        rule_tokens = rule_meta.get("compressed_tokens_est") or estimate_tokens_from_messages(
            rule_result, model=self.main_model
        )
        risk_flags: list[str] = []
        if policy.mode == "protected":
            risk_flags.append("protected_policy")
        if spans:
            risk_flags.append("protected_spans")

        route_info = None
        if self.route:
            active = self.active_option or self.route.cheap_options[0]
            route_info = {
                "main_model": self.main_model,
                "compressor": active.model,
                "candidate_count": len(self.route.cheap_options),
                "candidates": [option.model for option in self.route.cheap_options],
                "main_input_price": self.route.main_input_price,
                "main_output_price": self.route.main_output_price,
                "probe": (
                    {
                        "enabled": self.enable_model_probe,
                        "available": self.probe_result.available,
                        "source": self.probe_result.source,
                        "provider": self.probe_result.provider,
                        "models_seen": list(self.probe_result.models_seen),
                        "failure_reason": self.probe_result.failure_reason,
                    }
                    if self.probe_result else {"enabled": self.enable_model_probe}
                ),
            }

        estimated_raw_cost = None
        estimated_rule_cost = None
        if self.route:
            raw_output_tokens = int(original_tokens * DEFAULT_OUTPUT_RATIO)
            rule_output_tokens = int(rule_tokens * DEFAULT_OUTPUT_RATIO)
            estimated_raw_cost = estimate_cost(
                original_tokens,
                raw_output_tokens,
                self.route.main_input_price,
                self.route.main_output_price,
            )
            estimated_rule_cost = estimate_cost(
                rule_tokens,
                rule_output_tokens,
                self.route.main_input_price,
                self.route.main_output_price,
            )

        base = {
            "mode": "shadow",
            "enabled": True,
            "original_tokens": original_tokens,
            "rule_tokens": rule_tokens,
            "estimated_raw_cost": round(estimated_raw_cost, 8) if estimated_raw_cost is not None else None,
            "estimated_rule_cost": round(estimated_rule_cost, 8) if estimated_rule_cost is not None else None,
            "policy_mode": policy.mode,
            "policy_target_ratio": policy.target_ratio,
            "policy_reason": policy.reason,
            "protected_span_count": len(spans),
            "protected_span_kinds": sorted({span.kind for span in spans}),
            "protected_spans_sample": [
                {"kind": span.kind, "value": span.value, "occurrences": span.occurrences}
                for span in spans[:16]
            ],
            "route": route_info,
            "rule_compression": rule_meta,
            "risk_flags": risk_flags,
            "notes": "shadow mode does not modify the request and does not call cheap models",
        }

        if not self.is_configured or not self.route:
            return {
                **base,
                "would_call_smart": False,
                "would_use_rule_only": True,
                "would_fallback_reason": "missing_route_or_api_config",
                "reason": "无同平台廉价模型可用或缺少API配置；线上应走纯规则路径",
                "estimated_smart_tokens": None,
                "estimated_smart_cost": None,
                "estimated_savings_pct": None,
                "selected_candidate": None,
                "candidate_diagnostics": [],
            }

        if rule_tokens < self.min_rule_tokens_for_smart:
            projection = self._project_profit(rule_tokens)
            return {
                **base,
                "would_call_smart": False,
                "would_use_rule_only": True,
                "would_fallback_reason": "short_input_profit_guard",
                "reason": "规则压缩后输入过短，调用廉价模型的固定成本不划算",
                "estimated_smart_tokens": None,
                "estimated_smart_cost": projection.get("smart_total_cost"),
                "estimated_savings_pct": projection.get("savings_pct"),
                "selected_candidate": projection.get("candidate"),
                "candidate_diagnostics": [projection],
            }

        try:
            selected, projected, diagnostics = self._select_best_option(rule_tokens)
        except Exception as e:
            return {
                **base,
                "would_call_smart": False,
                "would_use_rule_only": True,
                "would_fallback_reason": "router_exception",
                "reason": f"候选路由器异常；线上应回退纯规则: {str(e)[:200]}",
                "estimated_smart_tokens": None,
                "estimated_smart_cost": None,
                "estimated_savings_pct": None,
                "selected_candidate": None,
                "candidate_diagnostics": [],
            }

        if selected is None or projected is None:
            all_context_blocked = bool(diagnostics) and all(
                item.get("blocked_by_context") for item in diagnostics
            )
            all_circuit_blocked = bool(diagnostics) and all(
                item.get("blocked_by_circuit") for item in diagnostics
            )
            fallback_reason = (
                "context_guard" if all_context_blocked
                else "circuit_breaker" if all_circuit_blocked
                else "profit_guard"
            )
            return {
                **base,
                "would_call_smart": False,
                "would_use_rule_only": True,
                "would_fallback_reason": fallback_reason,
                "reason": "shadow 预测不应调用廉价模型；线上应走纯规则路径",
                "estimated_smart_tokens": None,
                "estimated_smart_cost": None,
                "estimated_savings_pct": None,
                "selected_candidate": None,
                "candidate_diagnostics": diagnostics,
            }

        expected_ratio = self._expected_ratio_for(selected)
        estimated_smart_tokens = max(1, int(rule_tokens * expected_ratio))
        raw_to_smart_savings_pct = None
        if estimated_raw_cost and projected.get("smart_total_cost") is not None:
            raw_to_smart_savings_pct = (
                (estimated_raw_cost - float(projected["smart_total_cost"])) / estimated_raw_cost * 100
                if estimated_raw_cost > 0 else 0.0
            )
        return {
            **base,
            "would_call_smart": True,
            "would_use_rule_only": False,
            "would_fallback_reason": None,
            "reason": "shadow 预测 v5 smart compression 可启用；真实请求仍保持原路径不变",
            "estimated_smart_tokens": estimated_smart_tokens,
            "estimated_smart_cost": projected.get("smart_total_cost"),
            "estimated_savings_pct": projected.get("savings_pct"),
            "estimated_raw_to_smart_savings_pct": round(raw_to_smart_savings_pct, 2) if raw_to_smart_savings_pct is not None else None,
            "selected_candidate": selected.model,
            "candidate_diagnostics": diagnostics,
        }

    def _assess_policy(self, messages: list[dict[str, Any]]) -> CompressionPolicy:
        policy = assess_compression_policy(messages)
        if policy.mode == "safe":
            return CompressionPolicy(policy.mode, self.safe_target_ratio, policy.reason, policy.risk_score)
        if policy.mode == "extreme":
            return CompressionPolicy(policy.mode, self.extreme_target_ratio, policy.reason, policy.risk_score)
        return CompressionPolicy(policy.mode, self.protected_target_ratio, policy.reason, policy.risk_score)

    def _score_fidelity(
        self,
        original: list[dict[str, Any]],
        compressed: list[dict[str, Any]],
    ) -> FidelityReport:
        report = score_semantic_fidelity(original, compressed, policy=self.current_policy)
        hard_signal_count = sum(
            report.original_signal_count.get(name, 0)
            for name in ("numbers", "paths", "urls", "emails", "code_symbols")
        )
        if self.current_policy and self.current_policy.mode == "protected":
            threshold = self.protected_min_fidelity_score
        elif hard_signal_count == 0:
            threshold = min(self.min_fidelity_score, 0.55)
        else:
            threshold = self.min_fidelity_score
        return FidelityReport(
            score=report.score,
            passed=report.score >= threshold,
            threshold=threshold,
            coverage=report.coverage,
            missing=report.missing,
            original_signal_count=report.original_signal_count,
        )

    def _expected_ratio_for(self, option: CheapModelOption | None = None) -> float:
        """Learned expected compression ratio for cost projection.

        Base target comes from dual-threshold policy. Successful real calls can
        refine it, but protected mode never becomes more aggressive than its guard.
        """
        base_ratio = self.current_policy.target_ratio if self.current_policy else self.expected_smart_ratio
        if not self.learning_enabled or option is None:
            return base_ratio
        stats = self.learning_stats.get(option.model)
        if not stats or stats.avg_ratio is None:
            return base_ratio
        learned = min(0.95, max(0.05, stats.avg_ratio * 1.10))
        if self.current_policy and self.current_policy.mode == "protected":
            return max(base_ratio, learned)
        return min(base_ratio, learned)

    def _project_profit(
        self,
        rule_tokens: int,
        option: CheapModelOption | None = None,
    ) -> dict[str, float | bool | str]:
        assert self.route is not None
        expected_ratio = self._expected_ratio_for(option)
        expected_smart_tokens = max(1, int(rule_tokens * expected_ratio))
        projection = estimate_route_profit(
            self.route,
            rule_tokens,
            expected_smart_tokens,
            cheap_option=option,
        )
        projection["expected_smart_ratio"] = round(expected_ratio, 4)
        return projection

    def _select_best_option(
        self,
        rule_tokens: int,
    ) -> tuple[CheapModelOption | None, dict | None, list[dict]]:
        """Pick the most profitable cheap candidate that can fit the input."""
        assert self.route is not None
        diagnostics: list[dict] = []
        best_option: CheapModelOption | None = None
        best_projection: dict | None = None

        for option in self.route.cheap_options:
            stats = self.learning_stats.get(option.model)
            if stats and stats.disabled_until_call > self._compress_calls:
                diagnostics.append({
                    "candidate": option.model,
                    "blocked_by_circuit": True,
                    "disabled_until_call": stats.disabled_until_call,
                    "consecutive_failures": stats.consecutive_failures,
                    "last_reason": stats.last_reason,
                })
                continue

            if rule_tokens > option.max_context:
                diagnostics.append({
                    "candidate": option.model,
                    "blocked_by_context": True,
                    "cheap_max_context": option.max_context,
                })
                continue

            projection = self._project_profit(rule_tokens, option=option)
            projection = dict(projection)
            projection["blocked_by_context"] = False
            projection["blocked_by_circuit"] = False
            projection["cheap_max_context"] = option.max_context
            if stats:
                projection["learning"] = {
                    "attempts": stats.attempts,
                    "successes": stats.successes,
                    "failures": stats.failures,
                    "consecutive_failures": stats.consecutive_failures,
                    "avg_ratio": round(stats.avg_ratio, 4) if stats.avg_ratio is not None else None,
                    "avg_savings_pct": round(stats.avg_savings_pct, 2) if stats.avg_savings_pct is not None else None,
                }
            diagnostics.append(projection)

            if not projection["profitable"] or projection["savings_pct"] < self.min_profit_margin * 100:
                continue
            if best_projection is None or projection["savings"] > best_projection["savings"]:
                best_option = option
                best_projection = projection

        return best_option, best_projection, diagnostics

    def _record_learning(
        self,
        option: CheapModelOption,
        success: bool,
        reason: str,
        rule_tokens: int | None = None,
        final_tokens: int | None = None,
        actual_profit: dict | None = None,
    ) -> None:
        """Record feedback and open a circuit breaker after repeated failures.

        This prevents a broken/low-quality compressor from being called repeatedly
        and increasing total cost. The next calls fall back to rule-only until the
        cooldown expires.
        """
        if not self.learning_enabled:
            return

        stats = self.learning_stats.setdefault(option.model, CandidateLearningStats())
        stats.attempts += 1
        stats.last_reason = reason

        if success:
            stats.successes += 1
            stats.consecutive_failures = 0
            if rule_tokens and final_tokens is not None and rule_tokens > 0:
                stats.total_ratio += final_tokens / rule_tokens
            if actual_profit and isinstance(actual_profit.get("savings_pct"), (int, float)):
                stats.total_savings_pct += float(actual_profit["savings_pct"])
            return

        stats.failures += 1
        stats.consecutive_failures += 1
        if stats.consecutive_failures >= self.max_consecutive_failures:
            stats.disabled_until_call = self._compress_calls + self.circuit_breaker_cooldown

    def _meta(
        self,
        mode: str,
        reason: str,
        rule_meta: dict,
        projected: dict | None = None,
        actual_profit: dict | None = None,
        candidate_diagnostics: list[dict] | None = None,
        smart_compression: dict | None = None,
        fidelity_report: FidelityReport | None = None,
    ) -> dict:
        route_info = None
        if self.route:
            active = self.active_option or self.route.cheap_options[0]
            route_info = {
                "main_model": self.main_model,
                "compressor": self.compressor_model,
                "selected_candidate": active.model,
                "candidate_count": len(self.route.cheap_options),
                "candidates": [option.model for option in self.route.cheap_options],
                "cross_generation": active.cross_generation,
                "note": active.note,
                "main_input_price": self.route.main_input_price,
                "cheap_input_price": active.input_price,
                "cheap_max_context": active.max_context,
                "probe": (
                    {
                        "enabled": self.enable_model_probe,
                        "available": self.probe_result.available,
                        "source": self.probe_result.source,
                        "provider": self.probe_result.provider,
                        "models_seen": list(self.probe_result.models_seen),
                        "failure_reason": self.probe_result.failure_reason,
                    }
                    if self.probe_result else {"enabled": self.enable_model_probe}
                ),
            }
        return {
            "mode": mode,
            "compressor": self.compressor_model or "none",
            "reason": reason,
            "route": route_info,
            "token_estimator": "tiktoken_if_installed_else_multilingual_fallback",
            "compression_policy": (
                {
                    "mode": self.current_policy.mode,
                    "target_ratio": self.current_policy.target_ratio,
                    "reason": self.current_policy.reason,
                    "risk_score": self.current_policy.risk_score,
                }
                if self.current_policy else None
            ),
            "protected_spans": {
                "count": len(self.current_protected_spans),
                "items": [
                    {"kind": span.kind, "value": span.value, "occurrences": span.occurrences}
                    for span in self.current_protected_spans[:32]
                ],
            },
            "fidelity_guard": (
                {
                    "score": fidelity_report.score,
                    "passed": fidelity_report.passed,
                    "threshold": fidelity_report.threshold,
                    "coverage": fidelity_report.coverage,
                    "missing": fidelity_report.missing,
                    "original_signal_count": fidelity_report.original_signal_count,
                }
                if fidelity_report else None
            ),
            "profit_guard": {
                "min_profit_margin_pct": round(self.min_profit_margin * 100, 2),
                "min_rule_tokens_for_smart": self.min_rule_tokens_for_smart,
                "projected": projected,
                "actual": actual_profit,
                "candidate_diagnostics": candidate_diagnostics,
                "learning": {
                    model: {
                        "attempts": stats.attempts,
                        "successes": stats.successes,
                        "failures": stats.failures,
                        "consecutive_failures": stats.consecutive_failures,
                        "disabled_until_call": stats.disabled_until_call,
                        "avg_ratio": round(stats.avg_ratio, 4) if stats.avg_ratio is not None else None,
                        "avg_savings_pct": round(stats.avg_savings_pct, 2) if stats.avg_savings_pct is not None else None,
                        "last_reason": stats.last_reason,
                    }
                    for model, stats in self.learning_stats.items()
                },
            },
            "rule_compression": rule_meta,
            "smart_compression": smart_compression,
        }

    def _call_compressor(
        self,
        messages: list[dict[str, Any]],
        policy: CompressionPolicy | None = None,
    ) -> list[dict[str, Any]]:
        """Call the cheap model to compress messages.

        Same API key and base_url as the user's main model; only `model` changes.
        """
        policy_text = ""
        if policy is not None:
            policy_text = (
                f"压缩阀门: {policy.mode}; 目标比例: {policy.target_ratio:.0%}; "
                f"原因: {policy.reason}。请在不违反绝对边界的前提下尽量贴近目标。"
            )
        protected_text = format_protected_spans(self.current_protected_spans)
        instruction_text = "\n".join(part for part in (policy_text, protected_text) if part)
        payload = {
            "model": self.compressor_model,
            "messages": [
                {"role": "system", "content": COMPRESSION_SYSTEM_PROMPT},
                {"role": "user", "content": instruction_text + "\n" + json.dumps(messages, ensure_ascii=False)},
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        return self._parse_json(content)

    def _parse_json(self, content: str) -> list[dict[str, Any]]:
        """Parse JSON returned by the cheap compressor model."""
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        parsed = json.loads(content.strip())
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("返回结果不是非空列表")
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError("列表元素不是 dict")
            if "role" not in item or "content" not in item:
                raise ValueError("元素缺少 role 或 content 字段")
            if not isinstance(item["content"], str):
                raise ValueError("content 字段不是字符串")
        return parsed

    def _validate(self, compressed: Any, original: list[dict[str, Any]]) -> bool:
        """Validate compressed output before it can reach the main model."""
        if not isinstance(compressed, list) or not compressed:
            return False

        orig_t = estimate_tokens_from_messages(original, model=self.main_model)
        comp_t = estimate_tokens_from_messages(compressed, model=self.main_model)

        if comp_t > orig_t:
            return False
        if any(m.get("role") == "system" for m in original) and not any(
            m.get("role") == "system" for m in compressed
        ):
            return False
        if not any(m.get("role") == "user" for m in compressed):
            return False
        return True


# ══════════════════════════════════════════════════════════════════════════════
# StatisticalAnalyzer (Headroom SmartCrusher)
# ══════════════════════════════════════════════════════════════════════════════

import csv
import io
import statistics
from enum import Enum


class FieldType(Enum):
    """Inferred field semantic type from statistical analysis."""
    ID = "id"                 # High uniqueness (>0.9), preserve but can compress format
    SCORE = "score"           # Bounded numeric (0-100, 0-1, etc.)
    TEMPORAL = "temporal"     # Time/date related
    ERROR = "error"           # Error keywords detected — NEVER discard
    TEXT = "text"             # General text content
    NUMERIC = "numeric"       # General numeric
    BOOLEAN = "boolean"       # True/false values
    LIST = "list"             # Array/sequence of values
    UNKNOWN = "unknown"


class FieldAnalysis:
    """Analysis result for a single field or value collection."""

    def __init__(
        self,
        field_type: FieldType,
        confidence: float,
        uniqueness: float = 0.0,
        is_anomaly: bool = False,
        is_protected: bool = False,
        compression_hint: str = "",
    ):
        self.field_type = field_type
        self.confidence = confidence
        self.uniqueness = uniqueness
        self.is_anomaly = is_anomaly
        self.is_protected = is_protected
        self.compression_hint = compression_hint

    def __repr__(self) -> str:
        return (
            f"FieldAnalysis(type={self.field_type.value}, conf={self.confidence:.2f}, "
            f"unique={self.uniqueness:.2f}, anomaly={self.is_anomaly}, "
            f"protected={self.is_protected}, hint={self.compression_hint!r})"
        )


# Error keywords that mark content as protected (never discard)
ERROR_KEYWORDS = frozenset({
    "error", "exception", "traceback", "fatal", "critical", "panic",
    "failure", "crash", "segfault", "oom", "timeout", "refused",
    "denied", "forbidden", "unauthorized", "not found", "500", "502",
    "503", "404", "401", "403", "错误", "异常", "失败", "崩溃",
    "超时", "拒绝", "未找到", "权限",
})

# Score range patterns: values commonly between 0-100 or 0-1
_SCORE_PATTERNS = [
    re.compile(r'\b(?:score|rating|confidence|accuracy|precision|recall|f1|auc)\s*[:=]\s*[\d.]+', re.I),
    re.compile(r'\b\d+(?:\.\d+)?%'),  # percentages
]


class StatisticalAnalyzer:
    """Headroom SmartCrusher: infer field semantics from statistical distributions.

    Instead of relying on field names to determine how to compress, this analyzer
    examines the actual data distribution to infer what type of content a field
    contains, enabling smarter compression decisions.

    Key capabilities:
    1. Field type inference from data distribution
    2. Anomaly detection (>2σ outliers auto-preserved)
    3. Array first/last preservation
    4. Lossless-first: try CSV+Schema compression first, use if ≥30% savings
    """

    def __init__(self, anomaly_sigma: float = 2.0):
        """
        Args:
            anomaly_sigma: Number of standard deviations for anomaly detection.
        """
        self.anomaly_sigma = anomaly_sigma

    def analyze_values(self, values: list[Any]) -> FieldAnalysis:
        """Analyze a collection of values to infer field type and properties.

        Args:
            values: List of values to analyze.

        Returns:
            FieldAnalysis with inferred type and compression hints.
        """
        if not values:
            return FieldAnalysis(FieldType.UNKNOWN, confidence=0.0)

        str_values = [str(v) for v in values]
        unique_ratio = len(set(str_values)) / len(str_values)

        # Check for error content (MUST be protected)
        for sv in str_values:
            lower = sv.lower()
            if any(kw in lower for kw in ERROR_KEYWORDS):
                return FieldAnalysis(
                    FieldType.ERROR,
                    confidence=0.95,
                    uniqueness=unique_ratio,
                    is_protected=True,
                    compression_hint="error_content_never_discard",
                )

        # High uniqueness → ID field
        if unique_ratio > 0.9 and len(values) > 3:
            return FieldAnalysis(
                FieldType.ID,
                confidence=0.9,
                uniqueness=unique_ratio,
                compression_hint="id_field_compress_format",
            )

        # Check for numeric values
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                pass

        if len(numeric_values) > len(values) * 0.7:
            return self._analyze_numeric(numeric_values, str_values, unique_ratio)

        # Check for temporal patterns
        if self._has_temporal_pattern(str_values):
            return FieldAnalysis(
                FieldType.TEMPORAL,
                confidence=0.85,
                uniqueness=unique_ratio,
                compression_hint="temporal_can_normalize",
            )

        # Check for boolean-like values
        bool_values = {'true', 'false', '1', '0', 'yes', 'no', '是', '否'}
        lower_set = {sv.lower().strip() for sv in str_values}
        if lower_set.issubset(bool_values):
            return FieldAnalysis(
                FieldType.BOOLEAN,
                confidence=0.9,
                uniqueness=unique_ratio,
                compression_hint="boolean_compress_to_short",
            )

        # Default: text
        return FieldAnalysis(
            FieldType.TEXT,
            confidence=0.5,
            uniqueness=unique_ratio,
            compression_hint="text_standard_compress",
        )

    def _analyze_numeric(
        self, numeric_values: list[float], str_values: list[str], unique_ratio: float
    ) -> FieldAnalysis:
        """Analyze numeric values for type inference."""
        if len(numeric_values) < 2:
            return FieldAnalysis(
                FieldType.NUMERIC, confidence=0.6, uniqueness=unique_ratio,
                compression_hint="numeric_few_values",
            )

        min_val = min(numeric_values)
        max_val = max(numeric_values)

        # Check for percentage/score (0-100 or 0-1)
        if 0 <= min_val and max_val <= 100:
            # Check if they look like percentages
            has_pct = any('%' in sv for sv in str_values)
            has_score_pattern = any(
                p.search(' '.join(str_values[:10])) for p in _SCORE_PATTERNS
            )
            if has_pct or has_score_pattern or (0 <= min_val and max_val <= 1):
                return FieldAnalysis(
                    FieldType.SCORE,
                    confidence=0.85,
                    uniqueness=unique_ratio,
                    compression_hint="score_field_round_or_compress",
                )

        # Check for anomaly
        is_anomaly = False
        try:
            mean = statistics.mean(numeric_values)
            stdev = statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0
            if stdev > 0:
                # Check if any value deviates > anomaly_sigma from mean
                for v in numeric_values:
                    if abs(v - mean) > self.anomaly_sigma * stdev:
                        is_anomaly = True
                        break
        except statistics.StatisticsError:
            pass

        return FieldAnalysis(
            FieldType.NUMERIC,
            confidence=0.8,
            uniqueness=unique_ratio,
            is_anomaly=is_anomaly,
            compression_hint="numeric_standard_compress" + ("_has_anomaly" if is_anomaly else ""),
        )

    # Pre-compiled temporal patterns (Y4 fix: avoid re.compile on every call)
    _TEMPORAL_PATTERNS: list[re.Pattern] = [
        re.compile(r'\d{4}-\d{2}-\d{2}'),
        re.compile(r'\d{4}/\d{2}/\d{2}'),
        re.compile(r'\d{2}:\d{2}:\d{2}'),
        re.compile(r'\d{10,13}'),  # unix timestamps
        re.compile(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', re.I),
    ]

    def _has_temporal_pattern(self, str_values: list[str]) -> bool:
        """Check if values match common time/date patterns."""
        # Check at least 30% of values match temporal patterns
        match_count = 0
        sample = str_values[:min(20, len(str_values))]
        for sv in sample:
            if any(p.search(sv) for p in self._TEMPORAL_PATTERNS):
                match_count += 1
        return match_count >= len(sample) * 0.3

    def preserve_array_ends(self, arr: list[Any]) -> list[Any]:
        """Preserve first and last elements of an array for context.

        If array has ≤ 4 elements, return as-is.
        If > 4 elements, return [first, second, '...(N-4 omitted)...', second_to_last, last].

        Args:
            arr: Input array/list.

        Returns:
            Condensed array preserving ends.
        """
        if len(arr) <= 4:
            return arr

        return [
            arr[0],
            arr[1],
            f"...({len(arr) - 4} omitted)...",
            arr[-2],
            arr[-1],
        ]

    def try_lossless_csv_schema(
        self, data: list[dict[str, Any]], min_savings_ratio: float = 0.30
    ) -> tuple[str | None, dict]:
        """Attempt lossless CSV+Schema compression.

        If the data is a list of dicts with consistent keys, try to compress
        it into a CSV representation + schema definition. If savings ≥ 30%,
        return the compressed form.

        Args:
            data: List of dict records.
            min_savings_ratio: Minimum savings ratio to accept (default 30%).

        Returns:
            Tuple of (compressed_string_or_None, metadata).
            Returns None if savings are insufficient.
        """
        if not data or not isinstance(data[0], dict):
            return None, {"reason": "not_list_of_dicts", "savings": 0}

        # Get schema (all keys)
        all_keys = set()
        for record in data[:50]:  # sample first 50
            all_keys.update(record.keys())

        if not all_keys:
            return None, {"reason": "no_keys", "savings": 0}

        keys = sorted(all_keys)

        # Build CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(keys)
        for record in data:
            row = [record.get(k, '') for k in keys]
            writer.writerow(row)

        csv_text = output.getvalue()

        # Schema definition (compact)
        schema_parts = []
        for key in keys:
            sample_values = [str(record.get(key, '')) for record in data[:10]]
            analysis = self.analyze_values(sample_values)
            schema_parts.append(f"{key}:{analysis.field_type.value}")

        schema_str = "SCHEMA[" + "|".join(schema_parts) + "]"
        compressed = f"{schema_str}\n{csv_text}"

        # Calculate savings
        original_size = sum(len(json.dumps(r, ensure_ascii=False)) for r in data)
        compressed_size = len(compressed)

        if original_size == 0:
            return None, {"reason": "empty_data", "savings": 0}

        savings_ratio = 1.0 - (compressed_size / original_size)

        meta = {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "savings_ratio": round(savings_ratio, 4),
            "record_count": len(data),
            "field_count": len(keys),
            "schema": schema_str,
        }

        if savings_ratio >= min_savings_ratio:
            return compressed, meta
        return None, {"reason": "savings_below_threshold", **meta}

    def analyze_and_compress_text(
        self, text: str
    ) -> tuple[str, list[FieldAnalysis]]:
        """Full statistical analysis pipeline for a text block.

        Parses the text for structured data (JSON), applies statistical
        analysis, and returns optimized text with analysis metadata.

        Args:
            text: Input text that may contain JSON/structured data.

        Returns:
            Tuple of (optimized_text, field_analyses).
        """
        analyses: list[FieldAnalysis] = []

        # Try to parse as JSON
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            # Not JSON — analyze as plain text
            analysis = self.analyze_values([text])
            analyses.append(analysis)
            return text, analyses

        # If it's a list of dicts, try CSV+Schema
        if isinstance(data, list) and data and isinstance(data[0], dict):
            compressed, meta = self.try_lossless_csv_schema(data)
            if compressed is not None:
                # Analyze each field
                for key in sorted(data[0].keys()):
                    values = [r.get(key) for r in data if key in r]
                    analysis = self.analyze_values(values)
                    analyses.append(analysis)
                return compressed, analyses

        # If it's a dict, analyze each top-level key
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    analysis = self.analyze_values(
                        [str(v) for v in value[:50]]
                    )
                    # Mark as LIST type
                    analysis.field_type = FieldType.LIST
                elif isinstance(value, (int, float)):
                    analysis = self.analyze_values([value])
                else:
                    analysis = self.analyze_values([str(value)])
                analyses.append(analysis)

        return text, analyses

    def get_protected_values(self, analyses: list[FieldAnalysis]) -> list[str]:
        """Extract compression hints for protected fields.

        Args:
            analyses: List of FieldAnalysis results.

        Returns:
            List of compression hints for fields that should be protected.
        """
        return [
            a.compression_hint for a in analyses
            if a.is_protected or a.is_anomaly
        ]
