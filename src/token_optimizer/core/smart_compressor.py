"""L1 v5: SmartCompressor — 同 key 零配置智能压缩。

⚠️ 前置条件（必须全部满足才激活模型压缩，否则自动退化为 v4 纯规则）：
  1. main_model 在同平台能映射到一个更便宜的模型（见 _SIBLING_MAP）
  2. api_key 非空（用同一个 key 调廉价模型）
  3. base_url 非空（同一平台，同一 API endpoint）
  4. API 调用成功且输出通过校验

若任一条件不满足 → 纯规则压缩（v4，零成本）

💡 为什么用 Flash 压缩：
  Flash（$0.10/M）做智能压缩器，把内容压到 ~200K，然后交给用户的主模型（如 mimo-v2.5-pro $1.00/M）执行核心推理。
  Flash 只负责压缩，**最终解决问题的一直是用户的主模型**。

廉价模型自动路由表（同平台、同 key，零配置）：
  mimo-v2.5-pro  → mimo-v2-flash     (同 MiMo，跨代，Flash 仍在售)
  mimo-v2.5      → mimo-v2-flash     (同 MiMo，跨代，Flash 仍在售)
  deepseek-v4-pro → deepseek-v4-flash  (同 DeepSeek)
  qwen-max/+      → qwen-turbo        (同阿里云)
  gpt-4o/-turbo   → gpt-4o-mini       (同 OpenAI)
  claude-3-*     → claude-3-haiku     (同 Anthropic)

流程：
  用户原始请求 → [规则预压缩 v4（零成本）] → [廉价模型智能压缩（同 key）] → 用户主模型执行

成本示例（MiMo 平台，1M tokens）：
  Raw Pro:                  $1.00
  规则 v4（零成本）:         $0.69（省 31%）
  Flash 压缩 + Pro 执行:     $0.10（Flash）+ $0.20（Pro 处理 200K）= $0.30（省 70%）
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from token_optimizer.core.signal_noise import (
    InputCompressor,
    CompressionLevel,
)


# ══════════════════════════════════════════════════════════════════════════════
# 廉价模型映射表
# ══════════════════════════════════════════════════════════════════════════════
# 格式：(匹配主模型名的关键词, 廉价模型名)
# 映射规则：
#   - 同平台（同 base_url），同一个 API key
#   - 廉价模型调用的计费独立，与主模型互不干扰
#   - 注意：mimo-v2-flash 是跨代模型（V2 代），但同平台同 key，仍在售
#   - 新增平台只需在此追加一行

_SIBLING_MAP: list[tuple[str, str]] = [
    # ---------- MiMo 平台 ----------
    # flash 输入 $0.10/M，是 pro $1.00/M 的 1/10
    ("mimo-v2.5-pro", "mimo-v2-flash"),
    ("mimo-v2.5",     "mimo-v2-flash"),
    ("mimo-pro",      "mimo-v2-flash"),

    # ---------- DeepSeek 平台 ----------
    ("deepseek-v4-pro", "deepseek-v4-flash"),
    ("deepseek-pro",    "deepseek-v4-flash"),

    # ---------- 阿里云 / Qwen 平台 ----------
    ("qwen-max",  "qwen-turbo"),
    ("qwen-plus", "qwen-turbo"),

    # ---------- OpenAI 平台 ----------
    ("gpt-4o",       "gpt-4o-mini"),
    ("gpt-4-turbo",  "gpt-4o-mini"),

    # ---------- Anthropic 平台 ----------
    ("claude-3-opus",     "claude-3-haiku"),
    ("claude-3.5-sonnet", "claude-3-haiku"),
    # claude-4 系列如有便宜版可在此追加
]


def find_cheap_sibling(model: str) -> str | None:
    """查询主模型是否有同平台可供调用的廉价模型。

    Args:
        model: 用户正在使用的主模型名

    Returns:
        廉价模型的名称，或 None（无匹配 → 不使用模型压缩）
    """
    model_lower = model.lower().strip()
    for pattern, cheap in _SIBLING_MAP:
        if pattern in model_lower:
            return cheap
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 压缩 Prompt（模型无关，Flash / DeepSeek Flash / Qwen Turbo 都能用）
# ══════════════════════════════════════════════════════════════════════════════

COMPRESSION_SYSTEM_PROMPT = """你是一个精确的输入压缩器。压缩对话消息列表，最少token保留所有关键信息。

规则：
1. 保留最近2轮完整对话
2. 旧轮只保留：核心指令、关键结论、重要代码片段
3. 移除客套话、填充词、重复内容、过时调试信息
4. 错误信息只保留错误类型和位置
5. 代码只保留函数签名和关键逻辑，移除注释和空行
6. 工具输出只保留关键数据
7. 系统消息保持完整

输入：JSON数组，元素 {"role": "...", "content": "..."}
输出：仅输出压缩后的JSON数组，不要任何解释。"""


# ══════════════════════════════════════════════════════════════════════════════
# SmartCompressor
# ══════════════════════════════════════════════════════════════════════════════

class SmartCompressor:
    """零配置智能压缩器——前提是同平台有可供调用的廉价模型。

    设计原则：
      1. 用户零配置——自动从 main_model 推断同平台廉价模型
      2. 同 key 调用——廉价模型和主模型共享一个 API key，不需要额外配置
      3. 安全降级——任何异常退化为 v4 纯规则，不影响主流程
      4. 不硬凑——没有廉价模型时不降级用昂贵模型，直接走纯规则
      5. 廉价模型只做压缩，**最终解决问题的是用户的主模型**

    使用方式（一行初始化，自动检测廉价模型）：
        sc = SmartCompressor(
            main_model="mimo-v2.5-pro",  # 用户已有的主模型
            api_key="sk-xxx",            # 用户已有的 API key
            base_url="https://...",      # 用户已有的 endpoint
        )

    🔑 关键行为：
      - is_configured=True  → 有廉价模型可用，compress() 先规则再模型压缩
      - is_configured=False → 无廉价模型或无 key，compress() 纯规则

    成本收益：
      - MiMo 平台：Flash 输入 $0.10/M vs Pro $1.00/M，省 70%
      - DeepSeek 平台：Flash 输入 $0.14/M vs Pro $0.435/M，省 60%+
      - OpenAI 平台：Mini 输入 $0.15/M vs 4o $2.50/M，省 90%+
      - 无廉价模型时：纯规则 v4，零额外成本
    """

    def __init__(
        self,
        main_model: str = "",
        api_key: str = "",
        base_url: str = "",
        timeout: float = 30.0,
        level: CompressionLevel = CompressionLevel.AGGRESSIVE,
    ):
        self.level = level
        self.rule_compressor = InputCompressor(level=level)
        self.timeout = timeout
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else ""

        # 自动判断：同平台是否有廉价模型？
        sibling = find_cheap_sibling(main_model) if main_model else None
        self.compressor_model = sibling or ""
        # 激活条件：有廉价模型 + 有 key + 有 base_url，三者缺一不可
        self.is_configured = bool(sibling and api_key and base_url)

    def compress(
        self,
        messages: list[dict[str, Any]],
        system_text: str = "",
    ) -> tuple[list[dict[str, Any]], dict]:
        """压缩消息列表。

        流程：
          1. 规则预压缩 v4（始终执行，零成本）
          2. 如果有廉价模型 → 调用它做智能压缩
          3. 如果调用失败或校验不通过 → 退回到规则压缩结果

        Args:
            messages: 对话消息列表
            system_text: 系统提示词（用于跨轮去重）

        Returns:
            (compressed_messages, metadata)
            metadata.mode 标识使用了哪种策略：
              - "smart"           → 规则 + 廉价模型智能压缩
              - "rule_only"       → 仅规则（无廉价模型可用）
              - "rule_only_fallback" → 规则（廉价模型调用失败）
        """
        # ── Step 1: 规则预压缩（始终执行，零成本） ──
        rule_result, rule_meta = self.rule_compressor.compress_messages(
            messages, system_text=system_text
        )

        # ── Step 2: 判断是否满足模型压缩条件 ──
        if not self.is_configured:
            return rule_result, {
                "mode": "rule_only",
                "compressor": "none",
                "reason": "无廉价模型可用或缺少API配置",
                "rule_compression": rule_meta,
                "smart_compression": None,
            }

        # ── Step 3: 调用廉价模型做智能压缩 ──
        try:
            smart_result = self._call_compressor(rule_result)

            if self._validate(smart_result, messages):
                total_original = rule_meta.get("original_tokens_est", 0)
                final_tokens = sum(
                    max(1, len(m.get("content", "")) // 3)
                    for m in smart_result
                    if isinstance(m.get("content", ""), str)
                )
                flash_in = rule_meta.get("compressed_tokens_est", 0)
                flash_out = max(
                    1, len(json.dumps(smart_result, ensure_ascii=False)) // 3
                )

                return smart_result, {
                    "mode": "smart",
                    "compressor": self.compressor_model,
                    "reason": "激活条件满足：有同平台廉价模型 + key + base_url",
                    "rule_compression": rule_meta,
                    "smart_compression": {
                        "model": self.compressor_model,
                        "input_tokens": flash_in,
                        "output_tokens": flash_out,
                        "final_tokens": final_tokens,
                        "original_tokens": total_original,
                        "total_savings_pct": round(
                            (1 - final_tokens / max(1, total_original)) * 100, 1
                        ),
                    },
                }

            # 校验不通过 → 退回规则结果
            return rule_result, {
                "mode": "rule_only_fallback",
                "compressor": self.compressor_model,
                "reason": "廉价模型输出校验未通过（过长/丢失系统消息/缺失用户消息）",
                "rule_compression": rule_meta,
                "smart_compression": None,
            }

        except Exception as e:
            # API 调用失败 → 退回规则结果
            return rule_result, {
                "mode": "rule_only_fallback",
                "compressor": self.compressor_model,
                "reason": f"廉价模型调用失败: {str(e)[:200]}",
                "rule_compression": rule_meta,
                "smart_compression": None,
            }

    def _call_compressor(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """调用廉价模型进行压缩。

        使用与主模型相同的 API key 和 base_url，
        仅 model 参数切换为廉价模型。

        注意：压缩后的内容最终交给**用户的主模型**执行，
        廉价模型**只做预处理**，不参与正式推理。
        """
        payload = {
            "model": self.compressor_model,
            "messages": [
                {"role": "system", "content": COMPRESSION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(messages, ensure_ascii=False),
                },
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
        """解析模型返回的 JSON，去掉可能的 markdown 代码块包裹。"""
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
        """校验压缩结果的有效性。

        检查点：
          1. 是列表且非空
          2. 总 token 数不超过原文
          3. 如果原文有 system 消息，压缩结果也必须保留
          4. 必须至少有一条 user 消息
        """
        if not isinstance(compressed, list) or not compressed:
            return False

        orig_t = sum(
            max(1, len(m.get("content", "")) // 3)
            for m in original
            if isinstance(m.get("content", ""), str)
        )
        comp_t = sum(
            max(1, len(m.get("content", "")) // 3)
            for m in compressed
            if isinstance(m.get("content", ""), str)
        )

        if comp_t > orig_t:
            return False
        if any(m.get("role") == "system" for m in original) and not any(
            m.get("role") == "system" for m in compressed
        ):
            return False
        if not any(m.get("role") == "user" for m in compressed):
            return False
        return True
