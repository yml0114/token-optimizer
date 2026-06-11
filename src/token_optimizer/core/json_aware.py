"""JSON-Aware Compressor — Phase 2: 结构化JSON压缩。

核心思想：理解JSON结构，比纯文本截断更聪明。
- 键名缩写：长键名 → 短别名（description → d0, temperature → t1）
- 数值截断：长小数 → 2位精度（3.14159265 → 3.14）
- 结构保持：输出仍是合法JSON，可被下游LLM解析
- 可逆：键映射 + 精度信息存在 CompressionStore 中

预期收益（vs CCR 纯截断）：
- 键名密集JSON：压缩比额外 -10~30%
- 结构完整性：100%（仍是合法JSON）
"""

from __future__ import annotations

import json
import time
from typing import Any


class JsonAwareCompressor:
    """JSON结构化压缩器。

    工作流：
    1. 解析JSON
    2. 收集所有键名，生成缩写映射
    3. 截断长小数
    4. 应用映射，输出压缩JSON
    5. 存储映射表（用于可逆还原）

    Args:
        max_key_alias_len: 缩写键名最大长度（默认3字符）
        number_precision: 数值保留小数位数（默认2）
        min_key_len: 键名超过此长度才缩写（默认5）
    """

    def __init__(
        self,
        max_key_alias_len: int = 3,
        number_precision: int = 2,
        min_key_len: int = 5,
    ):
        self.max_key_alias_len = max_key_alias_len
        self.number_precision = number_precision
        self.min_key_len = min_key_len
        self._alias_counter = 0

    def _generate_alias(self) -> str:
        """生成短别名：a0, a1, ..., a9, b0, ..., z9, aa0, ..."""
        n = self._alias_counter
        self._alias_counter += 1
        if n < 260:  # a0-z9
            return chr(ord('a') + n // 10) + str(n % 10)
        else:
            return f"x{n}"

    def _collect_keys(self, obj: Any, keys: set) -> None:
        """递归收集所有键名"""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str):
                    keys.add(k)
                self._collect_keys(v, keys)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_keys(item, keys)

    def _build_key_mapping(self, keys: set) -> dict[str, str]:
        """生成键名缩写映射"""
        mapping = {}
        for key in sorted(keys):  # 排序保证一致性
            if len(key) > self.min_key_len:
                mapping[key] = self._generate_alias()
        return mapping

    def _truncate_numbers(self, obj: Any) -> Any:
        """递归截断数值"""
        if isinstance(obj, float):
            return round(obj, self.number_precision)
        elif isinstance(obj, dict):
            return {k: self._truncate_numbers(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._truncate_numbers(item) for item in obj]
        return obj

    def _apply_key_mapping(self, obj: Any, mapping: dict[str, str]) -> Any:
        """递归应用键名映射"""
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                new_key = mapping.get(k, k)
                new_dict[new_key] = self._apply_key_mapping(v, mapping)
            return new_dict
        elif isinstance(obj, list):
            return [self._apply_key_mapping(item, mapping) for item in obj]
        return obj

    def compress(self, text: str) -> tuple[str, dict]:
        """压缩JSON文本。

        Returns:
            (compressed_json_str, metadata)
            metadata 包含 key_mapping 和 number_precision 用于可逆还原
        """
        t0 = time.perf_counter()

        # 解析JSON
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text, {"skipped": True, "reason": "invalid_json"}

        # 收集键名
        keys = set()
        self._collect_keys(data, keys)

        # 构建映射
        self._alias_counter = 0
        key_mapping = self._build_key_mapping(keys)

        # 截断数值
        data = self._truncate_numbers(data)

        # 应用键名映射
        compressed_data = self._apply_key_mapping(data, key_mapping)

        # 序列化
        compressed = json.dumps(compressed_data, ensure_ascii=False, separators=(',', ':'))

        elapsed = (time.perf_counter() - t0) * 1000

        savings = len(text) - len(compressed)
        savings_pct = (savings / max(1, len(text))) * 100

        metadata = {
            "skipped": False,
            "key_mapping": key_mapping,
            "keys_abbreviated": len(key_mapping),
            "total_keys": len(keys),
            "number_precision": self.number_precision,
            "original_len": len(text),
            "compressed_len": len(compressed),
            "savings_chars": savings,
            "savings_pct": round(savings_pct, 1),
            "elapsed_ms": round(elapsed, 3),
        }

        return compressed, metadata

    def decompress(self, compressed_text: str, metadata: dict) -> str:
        """可逆还原（需要key_mapping）。

        Args:
            compressed_text: 压缩后的JSON文本
            metadata: compress() 返回的metadata

        Returns:
            还原后的JSON文本
        """
        if metadata.get("skipped"):
            return compressed_text

        key_mapping = metadata.get("key_mapping", {})
        # 反转映射：alias → original_key
        reverse_mapping = {v: k for k, v in key_mapping.items()}

        try:
            data = json.loads(compressed_text)
        except (json.JSONDecodeError, ValueError):
            return compressed_text

        # 反转键名
        restored = self._apply_key_mapping(data, reverse_mapping)

        # 注意：数值精度无法完美还原（lossy）
        # 但对于LLM场景，2位精度通常足够

        return json.dumps(restored, ensure_ascii=False, indent=2)
