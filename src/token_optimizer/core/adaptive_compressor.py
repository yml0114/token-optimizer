"""Adaptive Compressor — 自适应内容类型路由。

核心思想：不同内容类型用不同压缩策略。
- JSON/结构化数据 → 跳过IC，直接走CCR（IC对结构化数据无效）
- 自然语言对话 → IC去噪 + CCR可逆压缩
- 短消息 → 直接跳过所有压缩

这样避免IC在JSON上白烧12ms却0%压缩的尴尬。
"""

from __future__ import annotations

import json
from typing import Any

from token_optimizer.core.signal_noise import InputCompressor, CompressionLevel
from token_optimizer.core.compression_store import CompressionStore


class ContentType:
    """内容类型判断"""
    JSON = "json"
    DIALOG = "dialog"
    SHORT = "short"      # 太短，不值得压缩
    MIXED = "mixed"      # 混合内容（对话+JSON片段）

    # JSON判断阈值
    JSON_MIN_LENGTH = 100       # 最短JSON长度
    JSON_QUOTE_DENSITY = 0.02   # 引号密度阈值（引号数/总字符数）

    @staticmethod
    def detect(text: str) -> str:
        """检测内容类型"""
        stripped = text.strip()

        # 太短不压缩
        if len(stripped) < 30:
            return ContentType.SHORT

        # 明确JSON开头
        if stripped[0] in ('{', '['):
            try:
                json.loads(stripped)
                return ContentType.JSON
            except (json.JSONDecodeError, ValueError):
                pass

        # 高引号密度 = 可能是JSON
        quote_count = stripped.count('"') + stripped.count("'")
        if len(stripped) > ContentType.JSON_MIN_LENGTH:
            density = quote_count / len(stripped)
            if density > ContentType.JSON_QUOTE_DENSITY:
                # 再试一次，可能是嵌在消息里的JSON
                # 找第一个 { 或 [ 开始
                for start_char, end_char in [('{', '}'), ('[', ']')]:
                    start = stripped.find(start_char)
                    if start >= 0:
                        end = stripped.rfind(end_char)
                        if end > start:
                            candidate = stripped[start:end+1]
                            try:
                                json.loads(candidate)
                                return ContentType.JSON
                            except (json.JSONDecodeError, ValueError):
                                pass
                return ContentType.MIXED

        return ContentType.DIALOG


class AdaptiveCompressor:
    """自适应压缩器：根据内容类型选择最优压缩路径。

    路由逻辑：
    1. JSON → 直接走CCR（跳过IC，省12ms）
    2. DIALOG → IC去噪 + CCR可逆
    3. SHORT → 不压缩
    4. MIXED → IC + CCR（保守策略）

    预期效果：
    - JSON场景：延迟 12ms → 0ms，压缩比不变
    - 对话场景：效果不变（仍走IC+CCR）
    """

    def __init__(
        self,
        level: CompressionLevel = CompressionLevel.MODERATE,
        ccr_max_entries: int = 100,
        ccr_default_ttl: int = 600,
    ):
        self.level = level
        self.ic = InputCompressor(level=level)
        self.ccr_max_entries = ccr_max_entries
        self.ccr_default_ttl = ccr_default_ttl

    def compress(
        self,
        messages: list[dict[str, Any]],
        keep_ratio: float = 0.4,
    ) -> tuple[list[dict[str, Any]], dict]:
        """自适应压缩入口。

        Args:
            messages: 聊天消息列表
            keep_ratio: CCR保留比例（0.4 = 保留40%原文）

        Returns:
            (压缩后的消息列表, 元数据)
        """
        import time
        t0 = time.perf_counter()

        # 统计
        stats = {
            "json_skipped_ic": 0,
            "dialog_used_ic": 0,
            "short_skipped": 0,
            "mixed_used_ic": 0,
            "total_messages": len(messages),
        }

        # 自适应路由：先分类，再按类型分组走不同管线
        dialog_msgs = []   # 需要走IC的消息索引
        json_msgs = []     # 跳过IC的消息索引
        short_msgs = []    # 不压缩的消息索引

        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if not isinstance(content, str) or not content.strip():
                short_msgs.append(i)
                continue
            content_type = ContentType.detect(content)
            if content_type == ContentType.SHORT:
                short_msgs.append(i)
                stats["short_skipped"] += 1
            elif content_type == ContentType.JSON:
                json_msgs.append(i)
                stats["json_skipped_ic"] += 1
            elif content_type == ContentType.MIXED:
                dialog_msgs.append(i)
                stats["mixed_used_ic"] += 1
            else:
                dialog_msgs.append(i)
                stats["dialog_used_ic"] += 1

        # Step 1: 对话消息走 IC 去噪（与 IC+CCR 方案完全一致）
        dialog_original = [messages[i] for i in dialog_msgs]
        if dialog_original:
            denoised, _ = self.ic.compress_messages(dialog_original)
        else:
            denoised = []

        # Step 2: 所有需要压缩的消息统一走 CCR（批量处理，保证去重效率）
        store = CompressionStore(
            max_entries=self.ccr_max_entries,
            default_ttl=self.ccr_default_ttl,
        )
        final = [None] * len(messages)
        total_saved = 0

        # 先放 short 消息
        for i in short_msgs:
            final[i] = messages[i]

        # 对话消息：IC结果进CCR（与 IC+CCR 一致）
        for idx, msg_idx in enumerate(dialog_msgs):
            msg = denoised[idx] if idx < len(denoised) else messages[msg_idx]
            content = msg.get("content", "")
            if not content.strip():
                final[msg_idx] = msg
                continue
            keep_len = max(1, int(len(content) * keep_ratio))
            compressed_text = content[:keep_len]
            hash_key, annotated = store.store(content, compressed_text)
            saved = len(content) - len(annotated)
            if saved > 0:
                final[msg_idx] = {**msg, "content": annotated}
                total_saved += saved
            else:
                final[msg_idx] = msg

        # JSON消息：跳过IC，原始内容直接进CCR
        for msg_idx in json_msgs:
            msg = messages[msg_idx]
            content = msg.get("content", "")
            if not content.strip():
                final[msg_idx] = msg
                continue
            keep_len = max(1, int(len(content) * keep_ratio))
            compressed_text = content[:keep_len]
            hash_key, annotated = store.store(content, compressed_text)
            saved = len(content) - len(annotated)
            if saved > 0:
                final[msg_idx] = {**msg, "content": annotated}
                total_saved += saved
            else:
                final[msg_idx] = msg

        # 填充None（不应发生）
        for i in range(len(final)):
            if final[i] is None:
                final[i] = messages[i]

        elapsed = (time.perf_counter() - t0) * 1000

        # 最终token统计
        from token_optimizer.core.signal_noise import InputCompressor as IC
        orig_tokens = sum(max(1, len(m.get("content", "")) // 3) for m in messages)
        comp_tokens = sum(max(1, len(m.get("content", "")) // 3) for m in final)

        stats.update({
            "elapsed_ms": round(elapsed, 2),
            "original_tokens_est": orig_tokens,
            "compressed_tokens_est": comp_tokens,
            "compression_ratio": round(comp_tokens / max(1, orig_tokens), 3),
            "savings_pct": round((1 - comp_tokens / max(1, orig_tokens)) * 100, 1),
            "ccr_entries": len(store._store),
            "saved_chars": total_saved,
        })

        return final, stats
