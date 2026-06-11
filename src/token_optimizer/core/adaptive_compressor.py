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
from token_optimizer.core.json_aware import JsonAwareCompressor
from token_optimizer.core.near_dedup import NearDeduplicator


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
        json_aware: bool = True,
        near_dedup: bool = True,
    ):
        self.level = level
        self.ic = InputCompressor(level=level)
        self.ccr_max_entries = ccr_max_entries
        self.ccr_default_ttl = ccr_default_ttl
        self.json_aware = json_aware
        self.json_compressor = JsonAwareCompressor() if json_aware else None
        self.near_dedup = near_dedup
        self.deduplicator = NearDeduplicator() if near_dedup else None

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
            "near_dedup_merged": 0,
            "near_dedup_saved_chars": 0,
            "total_messages": len(messages),
        }

        # Phase 3: 近似去重（在所有压缩之前）
        if self.near_dedup and self.deduplicator:
            messages, dedup_stats = self.deduplicator.deduplicate(messages)
            stats["near_dedup_merged"] = dedup_stats.get("duplicates_found", 0)
            stats["near_dedup_saved_chars"] = dedup_stats.get("saved_chars", 0)

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

        # JSON消息：跳过IC，走JSON-aware压缩或CCR
        for msg_idx in json_msgs:
            msg = messages[msg_idx]
            content = msg.get("content", "")
            if not content.strip():
                final[msg_idx] = msg
                continue

            if self.json_aware and self.json_compressor:
                # Phase 2: JSON-aware预处理（键名缩写+数值截断），再进CCR全局压缩
                json_compressed, json_meta = self.json_compressor.compress(content)
                if not json_meta.get("skipped"):
                    ja_saved = json_meta.get("savings_chars", 0)
                    stats["json_aware_used"] = stats.get("json_aware_used", 0) + 1
                    stats["json_aware_saved_chars"] = stats.get("json_aware_saved_chars", 0) + ja_saved
                    # JsonAware结果送CCR做全局压缩（这才是压缩主力）
                    content_to_ccr = json_compressed
                else:
                    content_to_ccr = content
            else:
                content_to_ccr = content

            # CCR全局压缩
            keep_len = max(1, int(len(content_to_ccr) * keep_ratio))
            compressed_text = content_to_ccr[:keep_len]
            hash_key, annotated = store.store(content_to_ccr, compressed_text)
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
            "json_aware_used": stats.get("json_aware_used", 0),
            "route_summary": _build_route_summary(stats),
        })

        return final, stats


def _build_route_summary(stats: dict) -> str:
    """构建路由摘要字符串"""
    parts = []
    if stats.get("near_dedup_merged", 0) > 0:
        parts.append(f"去重:{stats['near_dedup_merged']}")
    if stats.get("json_skipped_ic", 0) > 0:
        ja = stats.get("json_aware_used", 0)
        if ja > 0:
            parts.append(f"JSON→JA+CCR:{ja}")
        else:
            parts.append(f"JSON跳过IC:{stats['json_skipped_ic']}")
    if stats.get("dialog_used_ic", 0) > 0:
        parts.append(f"对话走IC:{stats['dialog_used_ic']}")
    if stats.get("mixed_used_ic", 0) > 0:
        parts.append(f"混合走IC:{stats['mixed_used_ic']}")
    return ",".join(parts) if parts else "无压缩"
