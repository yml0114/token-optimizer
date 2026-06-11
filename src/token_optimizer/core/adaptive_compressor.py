"""Adaptive Compressor — 自适应内容类型路由 + 信息密度压缩。

核心思想：不同内容类型用不同压缩策略。
- JSON/结构化数据 → 跳过IC，直接走CCR（IC对结构化数据无效）
- 自然语言对话 → IC去噪 + 密度压缩（按信息密度选择保留内容）
- 短消息 → 直接跳过所有压缩

Phase 5: 密度压缩替代纯截断
- 评分信号：数值(+3)、百分比/货币(+2)、专有名词(+1.5)、KV结构(+2)
- 填充惩罚：问候语、确认语(-5)
- 保留高信息密度内容，删除低价值填充语
"""

from __future__ import annotations

import json
import re as _re
from typing import Any

from token_optimizer.core.signal_noise import InputCompressor, CompressionLevel
from token_optimizer.core.compression_store import CompressionStore
from token_optimizer.core.json_aware import JsonAwareCompressor
from token_optimizer.core.near_dedup import NearDeduplicator


# ── 信息密度压缩 ──────────────────────────────────────────────────────────

def _chunk_info_density(chunk: str) -> float:
    """Score a text chunk by information density. Higher = more valuable."""
    text = chunk.strip()
    if not text:
        return 0.0

    score = 0.0
    length = max(len(text), 1)

    # Numbers (high value): prices, counts, dates, percentages
    score += len(_re.findall(r'\b\d[\d,.]*\b', text)) * 3.0
    # Percentages and currency
    score += len(_re.findall(r'\d+[%％]', text)) * 2.0
    score += len(_re.findall(r'[$€¥£]\d', text)) * 2.0
    # Proper nouns / technical terms
    score += len(_re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)) * 1.5
    # Key-value patterns
    score += len(_re.findall(r'[\w]+[:：=→]', text)) * 2.0
    # Code/technical markers
    if any(ch in text for ch in ['{', '}', '()', '=>', '->', '```']):
        score += 3.0

    # Filler penalties
    if _re.match(r'^(ok|okay|sure|好的|收到|了解|noted|i see|got it|understood|'
                 r"i'm ready|ready|准备好了|thanks|thank you|谢谢)\s*[.!?。！？]?\s*$",
                 text, _re.IGNORECASE):
        score -= 5.0

    # Length bonus (normalized)
    score += min(length / 50, 3.0)

    return score / max(length ** 0.3, 1)


def _density_compress(text: str, target_len: int) -> str:
    """Information-density-aware compression.

    Score each sentence/chunk by information density and keep the most
    valuable ones up to target_len. Falls back to truncation if single chunk.
    """
    if target_len >= len(text):
        return text

    # Split into sentences or lines
    lines = text.split('\n')
    chunks = []
    for line in lines:
        if len(line) > 120:
            parts = _re.split(r'(?<=[.!?。！？])\s+', line)
            chunks.extend(parts)
        else:
            chunks.append(line)

    if len(chunks) <= 1:
        return text[:target_len]

    # Score and sort by density
    scored = [(_chunk_info_density(c), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Keep highest-value chunks up to target
    kept = []
    total_len = 0
    kept_ids = set()
    for score, chunk in scored:
        add_len = len(chunk) + (1 if kept else 0)
        if total_len + add_len <= target_len:
            kept.append(chunk)
            kept_ids.add(id(chunk))
            total_len += add_len

    if not kept:
        return text[:target_len]

    # Reconstruct in original order
    result = '\n'.join(c for c in chunks if id(c) in kept_ids)
    if len(result) > target_len:
        result = result[:target_len]
    return result


# ── 内容类型检测 ──────────────────────────────────────────────────────────

class ContentType:
    JSON = "json"
    DIALOG = "dialog"
    SHORT = "short"
    MIXED = "mixed"

    JSON_MIN_LENGTH = 100
    JSON_QUOTE_DENSITY = 0.02

    @staticmethod
    def detect(text: str) -> str:
        stripped = text.strip()
        if len(stripped) < 30:
            return ContentType.SHORT

        if stripped[0] in ('{', '['):
            try:
                json.loads(stripped)
                return ContentType.JSON
            except (json.JSONDecodeError, ValueError):
                pass

        quote_count = stripped.count('"') + stripped.count("'")
        if len(stripped) > ContentType.JSON_MIN_LENGTH:
            density = quote_count / len(stripped)
            if density > ContentType.JSON_QUOTE_DENSITY:
                for sc, ec in [('{', '}'), ('[', ']')]:
                    start = stripped.find(sc)
                    if start >= 0:
                        end = stripped.rfind(ec)
                        if end > start:
                            try:
                                json.loads(stripped[start:end+1])
                                return ContentType.JSON
                            except (json.JSONDecodeError, ValueError):
                                pass
                return ContentType.MIXED

        return ContentType.DIALOG


# ── 自适应压缩器 ──────────────────────────────────────────────────────────

class AdaptiveCompressor:
    """自适应压缩器：根据内容类型选择最优压缩路径。

    路由逻辑：
    1. JSON → JSON-aware + CCR（跳过IC）
    2. DIALOG → IC去噪 + 密度压缩
    3. SHORT → 不压缩
    4. MIXED → IC + 密度压缩
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

        stats = {
            "json_skipped_ic": 0, "dialog_used_ic": 0,
            "short_skipped": 0, "mixed_used_ic": 0,
            "near_dedup_merged": 0, "near_dedup_saved_chars": 0,
            "total_messages": len(messages),
        }

        # Phase 3: 近似去重
        if self.near_dedup and self.deduplicator:
            messages, dedup_stats = self.deduplicator.deduplicate(messages)
            stats["near_dedup_merged"] = dedup_stats.get("duplicates_found", 0)
            stats["near_dedup_saved_chars"] = dedup_stats.get("saved_chars", 0)

        # 自适应路由
        dialog_msgs, json_msgs, short_msgs = [], [], []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if not isinstance(content, str) or not content.strip():
                short_msgs.append(i)
                continue
            ct = ContentType.detect(content)
            if ct == ContentType.SHORT:
                short_msgs.append(i); stats["short_skipped"] += 1
            elif ct == ContentType.JSON:
                json_msgs.append(i); stats["json_skipped_ic"] += 1
            elif ct == ContentType.MIXED:
                dialog_msgs.append(i); stats["mixed_used_ic"] += 1
            else:
                dialog_msgs.append(i); stats["dialog_used_ic"] += 1

        # Step 1: 对话消息走 IC 去噪
        dialog_original = [messages[i] for i in dialog_msgs]
        denoised, _ = self.ic.compress_messages(dialog_original) if dialog_original else ([], {})

        # Step 2: CCR + 密度压缩
        store = CompressionStore(max_entries=self.ccr_max_entries, default_ttl=self.ccr_default_ttl)
        final = [None] * len(messages)
        total_saved = 0

        for i in short_msgs:
            final[i] = messages[i]

        # 对话消息：IC结果 + 密度压缩
        for idx, msg_idx in enumerate(dialog_msgs):
            msg = denoised[idx] if idx < len(denoised) else messages[msg_idx]
            content = msg.get("content", "")
            if not content.strip():
                final[msg_idx] = msg; continue
            keep_len = max(1, int(len(content) * keep_ratio))
            compressed_text = _density_compress(content, keep_len)
            hash_key, annotated = store.store(content, compressed_text)
            saved = len(content) - len(annotated)
            if saved > 0:
                final[msg_idx] = {**msg, "content": annotated}
                total_saved += saved
            else:
                final[msg_idx] = msg

        # JSON消息：JSON-aware + 密度压缩
        for msg_idx in json_msgs:
            msg = messages[msg_idx]
            content = msg.get("content", "")
            if not content.strip():
                final[msg_idx] = msg; continue

            content_to_ccr = content
            if self.json_aware and self.json_compressor:
                json_compressed, json_meta = self.json_compressor.compress(content)
                if not json_meta.get("skipped"):
                    stats["json_aware_used"] = stats.get("json_aware_used", 0) + 1
                    stats["json_aware_saved_chars"] = stats.get("json_aware_saved_chars", 0) + json_meta.get("savings_chars", 0)
                    content_to_ccr = json_compressed

            keep_len = max(1, int(len(content_to_ccr) * keep_ratio))
            compressed_text = _density_compress(content_to_ccr, keep_len)
            hash_key, annotated = store.store(content_to_ccr, compressed_text)
            saved = len(content) - len(annotated)
            if saved > 0:
                final[msg_idx] = {**msg, "content": annotated}
                total_saved += saved
            else:
                final[msg_idx] = msg

        # 填充None
        for i in range(len(final)):
            if final[i] is None:
                final[i] = messages[i]

        elapsed = (time.perf_counter() - t0) * 1000
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
    parts = []
    if stats.get("near_dedup_merged", 0) > 0:
        parts.append(f"去重:{stats['near_dedup_merged']}")
    if stats.get("json_skipped_ic", 0) > 0:
        ja = stats.get("json_aware_used", 0)
        parts.append(f"JSON:{stats['json_skipped_ic']}(JA:{ja})" if ja else f"JSON:{stats['json_skipped_ic']}")
    if stats.get("dialog_used_ic", 0) > 0:
        parts.append(f"对话:{stats['dialog_used_ic']}")
    if stats.get("mixed_used_ic", 0) > 0:
        parts.append(f"混合:{stats['mixed_used_ic']}")
    return ",".join(parts) if parts else "无压缩"
