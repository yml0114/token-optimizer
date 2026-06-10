"""L1 v5: SmartCompressor — Profit-Aware same-key smart compression.

前置条件（全部满足才激活模型压缩，否则自动退化为 v4 纯规则）：
  1. main_model 能映射到同平台廉价模型（见 ROUTES）
  2. api_key 非空（用同一个 key 调廉价模型）
  3. base_url 非空（同一平台，同一 API endpoint）
  4. 预计成本收益为正，且超过 min_profit_margin
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
from dataclasses import dataclass
from typing import Any

import httpx

from token_optimizer.core.signal_noise import (
    InputCompressor,
    CompressionLevel,
)


# ══════════════════════════════════════════════════════════════════════════════
# Pricing / Routing
# ══════════════════════════════════════════════════════════════════════════════

TOKENS_PER_MILLION = 1_000_000
DEFAULT_OUTPUT_RATIO = 0.20
DEFAULT_SMART_TARGET_RATIO = 0.35
DEFAULT_MIN_PROFIT_MARGIN = 0.05  # require at least 5% cheaper than rule-only


@dataclass(frozen=True)
class ModelRoute:
    """A same-platform cheap-model route.

    All prices are USD per 1M tokens.
    The cheap model shares the same API key and base_url with the main model.
    """

    pattern: str
    cheap_model: str
    main_input_price: float
    main_output_price: float
    cheap_input_price: float
    cheap_output_price: float
    cheap_max_context: int = 1_000_000
    cross_generation: bool = False
    note: str = ""


ROUTES: tuple[ModelRoute, ...] = (
    # MiMo V2.5 has no same-generation Flash. Use still-available V2 Flash.
    ModelRoute(
        pattern="mimo-v2.5-pro",
        cheap_model="mimo-v2-flash",
        main_input_price=1.00,
        main_output_price=3.00,
        cheap_input_price=0.10,
        cheap_output_price=0.30,
        cheap_max_context=256_000,
        cross_generation=True,
        note="MiMo V2.5 Pro → V2 Flash：跨代，但同平台同 key，Flash 仍可调用",
    ),
    ModelRoute(
        pattern="mimo-v2.5",
        cheap_model="mimo-v2-flash",
        main_input_price=0.14,
        main_output_price=0.28,
        cheap_input_price=0.10,
        cheap_output_price=0.30,
        cheap_max_context=256_000,
        cross_generation=True,
        note="MiMo V2.5 → V2 Flash：主模型本身已很便宜，必须先算账，收益不足则回退 v4",
    ),
    ModelRoute(
        pattern="mimo-pro",
        cheap_model="mimo-v2-flash",
        main_input_price=1.00,
        main_output_price=3.00,
        cheap_input_price=0.10,
        cheap_output_price=0.30,
        cheap_max_context=256_000,
        cross_generation=True,
        note="MiMo Pro alias → V2 Flash",
    ),

    # DeepSeek / Qwen / OpenAI / Anthropic examples. Prices are conservative defaults;
    # profit gate protects users if a route becomes uneconomical.
    ModelRoute("deepseek-v4-pro", "deepseek-v4-flash", 0.435, 0.87, 0.14, 0.28),
    ModelRoute("deepseek-pro", "deepseek-v4-flash", 0.435, 0.87, 0.14, 0.28),
    ModelRoute("qwen-max", "qwen-turbo", 2.40, 9.60, 0.05, 0.20),
    ModelRoute("qwen-plus", "qwen-turbo", 0.40, 1.20, 0.05, 0.20),
    ModelRoute("gpt-4o", "gpt-4o-mini", 2.50, 10.00, 0.15, 0.60),
    ModelRoute("gpt-4-turbo", "gpt-4o-mini", 10.00, 30.00, 0.15, 0.60),
    ModelRoute("claude-3-opus", "claude-3-haiku", 15.00, 75.00, 0.25, 1.25),
    ModelRoute("claude-3.5-sonnet", "claude-3-haiku", 3.00, 15.00, 0.25, 1.25),
)


def find_route(model: str) -> ModelRoute | None:
    """Find a same-platform cheap-model route for the user's main model."""
    model_lower = model.lower().strip()
    for route in ROUTES:
        if route.pattern in model_lower:
            return route
    return None


def find_cheap_sibling(model: str) -> str | None:
    """Backward-compatible helper: return cheap model name or None."""
    route = find_route(model)
    return route.cheap_model if route else None


def estimate_tokens_from_messages(messages: list[dict[str, Any]]) -> int:
    """Fast deterministic token estimate.

    This intentionally mirrors the existing project convention: ~3 chars/token.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += max(1, len(content) // 3)
    return total


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
) -> dict[str, float | bool]:
    """Compare rule-only cost vs cheap-compressor + main-model cost.

    Rule-only path:
        main_model(rule_tokens → answer)

    Smart path:
        cheap_model(rule_tokens → smart_tokens) + main_model(smart_tokens → answer)
    """
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
        route.cheap_input_price,
        route.cheap_output_price,
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
        "rule_cost": round(rule_cost, 8),
        "smart_total_cost": round(smart_total_cost, 8),
        "compressor_cost": round(compressor_cost, 8),
        "main_after_cost": round(main_after_cost, 8),
        "savings": round(savings, 8),
        "savings_pct": round(savings_pct, 2),
        "profitable": savings > 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Compression prompt
# ══════════════════════════════════════════════════════════════════════════════

COMPRESSION_SYSTEM_PROMPT = """你是一个精确的输入压缩器。压缩对话消息列表，最少token保留所有关键信息。

绝对边界：
1. 你只压缩，不解决用户问题
2. 不新增事实，不改写任务目标，不改变代码语义
3. system 消息必须完整保留
4. 最近 2 轮完整保留，旧轮只保留核心指令、关键结论、重要代码片段
5. 错误信息只保留错误类型、文件、行号、关键堆栈
6. 工具输出只保留可复现任务所需的关键数据
7. 如果压缩会导致信息丢失，宁可少压缩

输入：JSON数组，元素 {"role": "...", "content": "..."}
输出：仅输出压缩后的JSON数组，不要解释，不要 markdown。"""


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
    ):
        self.main_model = main_model
        self.level = level
        self.rule_compressor = InputCompressor(level=level)
        self.timeout = timeout
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.route = find_route(main_model) if main_model else None
        self.compressor_model = self.route.cheap_model if self.route else ""
        self.min_profit_margin = min_profit_margin
        self.min_rule_tokens_for_smart = min_rule_tokens_for_smart
        self.expected_smart_ratio = expected_smart_ratio

        self.is_configured = bool(self.route and api_key and base_url)

    def compress(
        self,
        messages: list[dict[str, Any]],
        system_text: str = "",
    ) -> tuple[list[dict[str, Any]], dict]:
        """Compress messages using rule-only or profit-aware smart compression."""
        rule_result, rule_meta = self.rule_compressor.compress_messages(
            messages, system_text=system_text
        )

        if not self.is_configured or not self.route:
            return rule_result, self._meta(
                mode="rule_only",
                reason="无同平台廉价模型可用或缺少API配置",
                rule_meta=rule_meta,
            )

        rule_tokens = rule_meta.get("compressed_tokens_est") or estimate_tokens_from_messages(rule_result)
        if rule_tokens < self.min_rule_tokens_for_smart:
            return rule_result, self._meta(
                mode="rule_only_profit_guard",
                reason="规则压缩后输入过短，调用廉价模型的固定成本不划算",
                rule_meta=rule_meta,
                projected=self._project_profit(rule_tokens),
            )

        if rule_tokens > self.route.cheap_max_context:
            return rule_result, self._meta(
                mode="rule_only_context_guard",
                reason="规则压缩后仍超过廉价模型上下文窗口，回退主模型纯规则路径",
                rule_meta=rule_meta,
                projected=None,
            )

        projected = self._project_profit(rule_tokens)
        if not projected["profitable"] or projected["savings_pct"] < self.min_profit_margin * 100:
            return rule_result, self._meta(
                mode="rule_only_profit_guard",
                reason="预测收益不足，不调用廉价模型",
                rule_meta=rule_meta,
                projected=projected,
            )

        try:
            smart_result = self._call_compressor(rule_result)

            if not self._validate(smart_result, messages):
                return rule_result, self._meta(
                    mode="rule_only_fallback",
                    reason="廉价模型输出校验未通过（过长/丢失system/缺失user）",
                    rule_meta=rule_meta,
                    projected=projected,
                )

            final_tokens = estimate_tokens_from_messages(smart_result)
            actual_profit = estimate_route_profit(self.route, rule_tokens, final_tokens)
            if (
                not actual_profit["profitable"]
                or actual_profit["savings_pct"] < self.min_profit_margin * 100
            ):
                return rule_result, self._meta(
                    mode="rule_only_profit_guard",
                    reason="实际压缩结果收益不足，回退纯规则",
                    rule_meta=rule_meta,
                    projected=projected,
                    actual_profit=actual_profit,
                )

            return smart_result, self._meta(
                mode="smart",
                reason="收益校验通过：廉价模型只压缩，主模型负责最终推理",
                rule_meta=rule_meta,
                projected=projected,
                actual_profit=actual_profit,
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
            return rule_result, self._meta(
                mode="rule_only_fallback",
                reason=f"廉价模型调用失败: {str(e)[:200]}",
                rule_meta=rule_meta,
                projected=projected,
            )

    def _project_profit(self, rule_tokens: int) -> dict[str, float | bool]:
        assert self.route is not None
        expected_smart_tokens = max(1, int(rule_tokens * self.expected_smart_ratio))
        return estimate_route_profit(self.route, rule_tokens, expected_smart_tokens)

    def _meta(
        self,
        mode: str,
        reason: str,
        rule_meta: dict,
        projected: dict | None = None,
        actual_profit: dict | None = None,
        smart_compression: dict | None = None,
    ) -> dict:
        route_info = None
        if self.route:
            route_info = {
                "main_model": self.main_model,
                "compressor": self.compressor_model,
                "cross_generation": self.route.cross_generation,
                "note": self.route.note,
                "main_input_price": self.route.main_input_price,
                "cheap_input_price": self.route.cheap_input_price,
                "cheap_max_context": self.route.cheap_max_context,
            }
        return {
            "mode": mode,
            "compressor": self.compressor_model or "none",
            "reason": reason,
            "route": route_info,
            "profit_guard": {
                "min_profit_margin_pct": round(self.min_profit_margin * 100, 2),
                "min_rule_tokens_for_smart": self.min_rule_tokens_for_smart,
                "projected": projected,
                "actual": actual_profit,
            },
            "rule_compression": rule_meta,
            "smart_compression": smart_compression,
        }

    def _call_compressor(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Call the cheap model to compress messages.

        Same API key and base_url as the user's main model; only `model` changes.
        """
        payload = {
            "model": self.compressor_model,
            "messages": [
                {"role": "system", "content": COMPRESSION_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(messages, ensure_ascii=False)},
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

        orig_t = estimate_tokens_from_messages(original)
        comp_t = estimate_tokens_from_messages(compressed)

        if comp_t > orig_t:
            return False
        if any(m.get("role") == "system" for m in original) and not any(
            m.get("role") == "system" for m in compressed
        ):
            return False
        if not any(m.get("role") == "user" for m in compressed):
            return False
        return True
