"""Near-Deduplicator — Phase 3: 近似去重压缩。

核心思想：对话历史中经常有"几乎相同"的消息（如重复的系统提示、
相似的工具输出、反复出现的代码模板）。检测并合并这些近似副本，
用短引用标记替代原文。

算法：字符三元组(trigram) + Jaccard相似度
- 将每条消息转为trigram集合
- 用滑动窗口与最近N条参考消息比较
- 相似度超过阈值 → 用短引用标记替代

预期收益：
- 重复系统提示场景：压缩比额外 -15~25%
- 多工具调用场景：压缩比额外 -5~15%
- 普通对话：无明显变化（消息太独特）

延迟开销：trigram计算 ~0.05ms/msg，Jaccard比较 ~0.01ms/pair
"""

from __future__ import annotations

import hashlib
from typing import Any


class NearDeduplicator:
    """近似去重器。

    Args:
        similarity_threshold: Jaccard相似度阈值（0.7 = 70%重叠视为近似副本）
        window_size: 滑动窗口大小（只与最近N条参考比较）
        min_content_length: 短于此长度的消息跳过去重
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        window_size: int = 20,
        min_content_length: int = 30,
        debug: bool = False,
    ):
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.min_content_length = min_content_length
        self.debug = debug

    @staticmethod
    def _to_trigrams(text: str) -> set[str]:
        """将文本转为字符三元组集合"""
        if len(text) < 3:
            return {text}
        return {text[i:i+3] for i in range(len(text) - 2)}

    @staticmethod
    def _content_hash(text: str) -> str:
        """生成短内容哈希（6字符hex，用于引用标记）"""
        return hashlib.md5(text.encode()).hexdigest()[:6]

    def _jaccard(self, set_a: set, set_b: set) -> float:
        """计算Jaccard相似度"""
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def deduplicate(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict]:
        """对消息列表进行近似去重。

        Returns:
            (处理后的消息列表, 统计信息)
            近似副本被替换为: [REF:{hash}]（短引用标记）
            原文只在首次出现时保留
        """
        stats = {
            "total_messages": len(messages),
            "duplicates_found": 0,
            "saved_chars": 0,
            "ref_entries": 0,
        }

        if not messages:
            return messages, stats

        # 参考窗口：(trigram_set, content_hash, original_length)
        ref_window: list[tuple[set[str], str, int]] = []
        result = []

        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < self.min_content_length:
                # 短消息不去重
                result.append(msg)
                continue

            trigrams = self._to_trigrams(content)
            content_hash = self._content_hash(content)

            # 在滑动窗口中查找近似副本
            best_match = None
            best_similarity = 0.0

            for ref_trigrams, ref_hash, ref_len in ref_window:
                sim = self._jaccard(trigrams, ref_trigrams)
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = (ref_trigrams, ref_hash, ref_len)

            if best_match and best_similarity >= self.similarity_threshold:
                # 近似副本 → 用短引用替代
                ref_hash = best_match[1]
                ref_marker = f"[REF:{ref_hash}]"
                saved = len(content) - len(ref_marker)
                result.append({**msg, "content": ref_marker})
                stats["duplicates_found"] += 1
                stats["saved_chars"] += saved
                if self.debug:
                    print(f"  DEDUP: sim={best_similarity:.3f} saved={saved} chars | {content[:60]}...")
                # 不把副本加入参考窗口（避免链式引用）
            else:
                # 唯一消息 → 保留原文，加入参考窗口
                result.append(msg)
                ref_window.append((trigrams, content_hash, len(content)))
                stats["ref_entries"] += 1
                if self.debug and best_similarity > 0:
                    print(f"  KEEP:  sim={best_similarity:.3f} (threshold={self.similarity_threshold}) | {content[:60]}...")
                # 窗口滑动
                if len(ref_window) > self.window_size:
                    ref_window.pop(0)

        return result, stats


class SimHashDeduplicator:
    """SimHash版本去重器（适合更长的消息）。

    比Jaccard更快（O(1)比较），但精度略低。
    适合消息平均长度 > 500字符的场景。
    """

    def __init__(
        self,
        hamming_threshold: int = 6,  # Hamming距离阈值（64位中最多6位不同）
        window_size: int = 20,
        min_content_length: int = 30,
    ):
        self.hamming_threshold = hamming_threshold
        self.window_size = window_size
        self.min_content_length = min_content_length

    @staticmethod
    def _to_trigrams(text: str) -> list[str]:
        if len(text) < 3:
            return [text]
        return [text[i:i+3] for i in range(len(text) - 2)]

    @staticmethod
    def _simhash(trigrams: list[str]) -> int:
        """计算SimHash值（64位）"""
        v = [0] * 64
        for tg in trigrams:
            h = int(hashlib.md5(tg.encode()).hexdigest()[:16], 16)
            for i in range(64):
                if (h >> i) & 1:
                    v[i] += 1
                else:
                    v[i] -= 1
        fingerprint = 0
        for i in range(64):
            if v[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint

    @staticmethod
    def _hamming_distance(a: int, b: int) -> int:
        """计算两个64位整数的Hamming距离"""
        return bin(a ^ b).count('1')

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:6]

    def deduplicate(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict]:
        stats = {
            "total_messages": len(messages),
            "duplicates_found": 0,
            "saved_chars": 0,
            "ref_entries": 0,
        }

        if not messages:
            return messages, stats

        # 参考窗口：(simhash, content_hash, original_length)
        ref_window: list[tuple[int, str, int]] = []
        result = []

        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < self.min_content_length:
                result.append(msg)
                continue

            trigrams = self._to_trigrams(content)
            fingerprint = self._simhash(trigrams)
            content_hash = self._content_hash(content)

            # 在窗口中找最近似
            best_match = None
            best_distance = 65  # 最大Hamming距离+1

            for ref_fp, ref_hash, ref_len in ref_window:
                dist = self._hamming_distance(fingerprint, ref_fp)
                if dist < best_distance:
                    best_distance = dist
                    best_match = (ref_fp, ref_hash, ref_len)

            if best_match and best_distance <= self.hamming_threshold:
                ref_hash = best_match[1]
                ref_marker = f"[REF:{ref_hash}]"
                saved = len(content) - len(ref_marker)
                result.append({**msg, "content": ref_marker})
                stats["duplicates_found"] += 1
                stats["saved_chars"] += saved
            else:
                result.append(msg)
                ref_window.append((fingerprint, content_hash, len(content)))
                stats["ref_entries"] += 1
                if len(ref_window) > self.window_size:
                    ref_window.pop(0)

        return result, stats
