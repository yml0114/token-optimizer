"""L1 v5: SmartCompressor — Flash-powered intelligent message compression.

Architecture:
  Rule-based pre-filter (v4) removes obvious noise at ZERO cost
  → Flash API ($0.10/M input) intelligently compresses remaining text
  → Output fed to main model (Pro at $1.00/M)

Cost math (MiMo-V2.5-Pro $1.00/M):
  Without: 1M tokens × $1.00 = $1.00
  With v5:  1M tokens × $0.10 (Flash) + 200K × $0.30 (Flash out)
            + 200K × $1.00 (Pro input) = $0.10 + $0.06 + $0.20 = $0.36
  Net savings: 64%

With L0 prefix cache:
  Flash: $0.16 + Pro cached: 200K × $0.20 = $0.04 → total $0.20
  Savings vs raw Pro: 80%

Flash is used ONLY for compression classification — its output is structured,
deterministic, and doesn't need Pro-level intelligence.

Key design decisions:
  1. Flash sees the FULL conversation, outputs a compressed version
  2. Compression prompt is engineered for maximum token reduction
  3. Rule-based pre-filter removes ~53% before Flash even sees it
  4. Flash compresses the remaining 47% to ~20% → net ~10% of original
  5. Graceful fallback: if Flash API fails, use rule-only compression
"""

from __future__ import annotations

import json
import re
from typing import Any

from token_optimizer.core.signal_noise import (
    InputCompressor,
    CompressionLevel,
)


# ══════════════════════════════════════════════════════════════════════════════
# Flash Compression Prompt
# ══════════════════════════════════════════════════════════════════════════════

COMPRESSION_PROMPT = """你是一个精确的输入压缩器。你的任务是压缩对话消息列表，使其占用最少的token，同时保留所有关键信息。

规则：
1. 保留最近2轮完整对话（user+assistant）
2. 旧轮对话只保留：核心指令、关键结论、重要代码片段
3. 移除所有客套话、填充词、重复内容、已过时的调试信息
4. 错误信息只保留错误类型和位置，移除完整stack trace
5. 代码片段只保留函数签名和关键逻辑，移除注释和空行
6. 工具输出只保留关键数据，移除格式化元数据
7. 系统消息保持完整不压缩

输入格式：JSON数组，每个元素 {"role": "user/assistant/tool/system", "content": "..."}
输出格式：JSON数组，同样的角色标签，但内容已压缩。只输出JSON，不要任何解释。

示例输入：
[
  {"role": "user", "content": "你好，我想请你帮我写一个函数"},
  {"role": "assistant", "content": "好的！我很乐意帮助你。请问你需要什么功能的函数呢？"},
  {"role": "user", "content": "就是那个排序函数，能把数组从小到大排"},
  {"role": "assistant", "content": "好的，这是一个快速排序的实现：\n```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n```\n这个函数使用快速排序算法，时间复杂度O(n log n)。"},
  {"role": "user", "content": "太好了！顺便问一下，你吃了吗？另外，我想改成降序排列"},
  {"role": "assistant", "content": "哈，吃了吃了！降序排列只需要改一下比较方向：\n```python\ndef quicksort_desc(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x > pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x < pivot]\n    return quicksort_desc(left) + middle + quicksort_desc(right)\n```"}
]

示例输出：
[
  {"role": "user", "content": "写一个降序排序函数"},
  {"role": "assistant", "content": "```python\ndef quicksort_desc(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x > pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x < pivot]\n    return quicksort_desc(left) + middle + quicksort_desc(right)\n```"}
]

注意：
- 输出必须是合法的JSON数组
- 每个元素必须有role和content字段
- 不要输出任何解释文字，只输出JSON
- 压缩比目标：减少60-80%的token数"""


class SmartCompressor:
    """Flash-powered intelligent message compressor.
    
    Uses MiMo-V2.5-Flash ($0.10/M input) to intelligently compress
    messages that remain after rule-based pre-filtering.
    
    Fallback: if Flash API fails, returns original messages unchanged.
    """
    
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.xiaomimimo.com/v1",
        model: str = "mimo-v2-flash",
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        
        # Also have rule-based compressor as pre-filter
        self.rule_compressor = InputCompressor(level=CompressionLevel.AGGRESSIVE)
    
    def compress(
        self,
        messages: list[dict[str, Any]],
        system_text: str = "",
    ) -> tuple[list[dict[str, Any]], dict]:
        """Smart compress: rules first, then Flash.
        
        Returns:
            (compressed_messages, metadata)
        """
        # ── Step 1: Rule-based pre-filter (zero cost) ──
        rule_compressed, rule_meta = self.rule_compressor.compress_messages(
            messages, system_text=system_text
        )
        
        # ── Step 2: Flash smart compression ──
        if not self.api_key:
            # No API key → rule-only fallback
            return rule_compressed, {
                "mode": "rule_only",
                "rule_compression": rule_meta,
                "smart_compression": {"skipped": True, "reason": "no_api_key"},
            }
        
        try:
            flash_result = self._flash_compress(rule_compressed)
            flash_tokens_in = rule_meta.get("compressed_tokens_est", 0)
            flash_tokens_out = max(1, len(str(flash_result)) // 3)
            
            # ── Step 3: Validate Flash output ──
            if self._validate_output(flash_result, messages):
                total_original = rule_meta.get("original_tokens_est", 0)
                final_tokens = sum(
                    max(1, len(m.get("content", "")) // 3)
                    for m in flash_result
                    if isinstance(m.get("content", ""), str)
                )
                
                return flash_result, {
                    "mode": "smart",
                    "rule_compression": rule_meta,
                    "smart_compression": {
                        "model": self.model,
                        "flash_input_tokens": flash_tokens_in,
                        "flash_output_tokens": flash_tokens_out,
                        "final_tokens": final_tokens,
                        "original_tokens": total_original,
                        "total_savings_pct": round(
                            (1 - final_tokens / max(1, total_original)) * 100, 1
                        ),
                    },
                }
            else:
                # Flash output invalid → rule-only fallback
                return rule_compressed, {
                    "mode": "rule_only_fallback",
                    "rule_compression": rule_meta,
                    "smart_compression": {"skipped": True, "reason": "validation_failed"},
                }
                
        except Exception as e:
            # Flash API error → rule-only fallback
            return rule_compressed, {
                "mode": "rule_only_fallback",
                "rule_compression": rule_meta,
                "smart_compression": {"skipped": True, "reason": str(e)},
            }
    
    def _flash_compress(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Call Flash API to compress messages."""
        import httpx
        
        # Build the request
        messages_json = json.dumps(messages, ensure_ascii=False, indent=2)
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": COMPRESSION_PROMPT},
                {"role": "user", "content": messages_json},
            ],
            "temperature": 0.0,  # Deterministic compression
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
        
        # Extract content
        content = data["choices"][0]["message"]["content"]
        
        # Parse JSON from response
        compressed = self._parse_json_response(content)
        return compressed
    
    def _parse_json_response(self, content: str) -> list[dict[str, Any]]:
        """Parse JSON from Flash response, handling markdown code blocks."""
        # Strip markdown code block wrapping
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Parse
        parsed = json.loads(content)
        
        # Validate structure
        if not isinstance(parsed, list):
            raise ValueError("Flash output is not a list")
        
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError("Flash output item is not a dict")
            if "role" not in item or "content" not in item:
                raise ValueError("Flash output item missing role/content")
            if not isinstance(item["content"], str):
                raise ValueError("Flash output content is not a string")
        
        return parsed
    
    def _validate_output(
        self,
        compressed: list[dict[str, Any]] | Any,
        original: list[dict[str, Any]],
    ) -> bool:
        """Validate Flash output is reasonable."""
        # Must be a list
        if not isinstance(compressed, list):
            return False
        
        # Must have at least 1 message
        if not compressed:
            return False
        
        # Must not be longer than original
        orig_tokens = sum(
            max(1, len(m.get("content", "")) // 3)
            for m in original
            if isinstance(m.get("content", ""), str)
        )
        comp_tokens = sum(
            max(1, len(m.get("content", "")) // 3)
            for m in compressed
            if isinstance(m.get("content", ""), str)
        )
        
        if comp_tokens > orig_tokens:
            return False
        
        # Must preserve system message if original had one
        has_system_original = any(m.get("role") == "system" for m in original)
        has_system_compressed = any(m.get("role") == "system" for m in compressed)
        if has_system_original and not has_system_compressed:
            return False
        
        # Must have at least one user message
        has_user = any(m.get("role") == "user" for m in compressed)
        if not has_user:
            return False
        
        return True
