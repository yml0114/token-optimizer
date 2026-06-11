"""
核心思想：不同内容类型用不同压缩策略。
- 自然语言对话 → IC去噪 + 密度压缩（按信息密度选择保留内容）
- 短消息 → 直接跳过所有压缩

Phase 5: 密度压缩替代纯截断
- 评分信号：数值(+3)、百分比/货币(+2)、专有名词(+1.5)、KV结构(+2)
- 填充惩罚：问候语、确认语(-5)
- 保留高信息密度内容，删除低价值填充语
- 代码保护：代码块（缩进/反引号/函数定义）整块保留，不逐行拆分
"""

from __future__ import annotations

import json
import re as _re
from typing import Any, List, Tuple

from token_optimizer.core.signal_noise import InputCompressor, CompressionLevel
from token_optimizer.core.near_dedup import NearDeduplicator


# ── 代码检测 ──────────────────────────────────────────────────────────────

def _looks_like_code(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _re.match(r'^(def |class |if |elif |else:|for |while |return |import |from |try|except|raise |with )', stripped):
        return True
    if _re.match(r'^[a-zA-Z_]\w*\s*[\+\-\*/]?=', stripped) and not stripped.endswith(('.', '!', '?')):
        return True
    if _re.match(r'^[a-zA-Z_]\w*\(', stripped):
        return True
    if any(ch in stripped for ch in ['{', '}', '->', '=>', '```']):
        return True
    if line.startswith('    ') and _re.search(r'[a-zA-Z_]\w*\s*[\(=]', stripped):
        return True
    return False


def _is_code_block(chunks: List[str]) -> bool:
    if not chunks:
        return False
    code_lines = sum(1 for c in chunks if _looks_like_code(c))
    return code_lines / len(chunks) > 0.4


def _merge_code_chunks(chunks: List[str]) -> List[Tuple[str, float, str]]:
    result = []
    code_buf = []
    for chunk in chunks:
        if _looks_like_code(chunk):
            code_buf.append(chunk)
        else:
            if code_buf:
                merged = '\n'.join(code_buf)
                score = _chunk_info_density(merged) * 2.0
                result.append((merged, score, 'code'))
                code_buf = []
            score = _chunk_info_density(chunk)
            result.append((chunk, score, 'text'))
    if code_buf:
        merged = '\n'.join(code_buf)
        score = _chunk_info_density(merged) * 2.0
        result.append((merged, score, 'code'))
    return result


# ── 信息密度压缩 ──────────────────────────────────────────────────────────

def _chunk_info_density(chunk: str) -> float:
    text = chunk.strip()
    if not text:
        return 0.0
    score = 0.0
    length = max(len(text), 1)
    score += len(_re.findall(r'\b\d[\d,.]*\b', text)) * 3.0
    score += len(_re.findall(r'\d+[%％]', text)) * 2.0
    score += len(_re.findall(r'[$€¥£]\d', text)) * 2.0
    score += len(_re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)) * 1.5
    score += len(_re.findall(r'[\w]+[:：=→]', text)) * 2.0
    if any(ch in text for ch in ['{', '}', '()', '=>', '->', '```']):
        score += 3.0
    if _re.match(r'^(ok|okay|sure|好的|收到|了解|noted|i see|got it|understood|'
                 r"i'm ready|ready|准备好了|thanks|thank you|谢谢)\s*[.!?。！？]?\s*$",
                 text, _re.IGNORECASE):
        score -= 5.0
    score += min(length / 50, 3.0)
    return score / max(length ** 0.3, 1)


def _density_compress(text: str, target_len: int) -> str:
    if target_len >= len(text):
        return text
    lines = text.split('\n')
    raw_chunks = []
    for line in lines:
        if len(line) > 120:
            parts = _re.split(r'(?<=[.!?。！？])\s+', line)
            raw_chunks.extend(parts)
        else:
            raw_chunks.append(line)
    if len(raw_chunks) <= 1:
        return text[:target_len]
    if _is_code_block(raw_chunks):
        scored_chunks = _merge_code_chunks(raw_chunks)
    else:
        scored_chunks = [(c, _chunk_info_density(c), 'text') for c in raw_chunks]
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    kept_content = set()
    total_len = 0
    for content, score, ctype in scored_chunks:
        add_len = len(content) + (1 if kept_content else 0)
        if total_len + add_len <= target_len:
            kept_content.add(content)
            total_len += add_len
        elif ctype == 'code' and len(content) <= target_len * 0.5:
            kept_content.add(content)
            total_len += add_len
    if not kept_content:
        return text[:target_len]
    result_parts = []
    for chunk in raw_chunks:
        if chunk in kept_content:
            result_parts.append(chunk)
        else:
            for content, _, ctype in scored_chunks:
                if ctype == 'code' and content in kept_content and chunk in content:
                    if content not in result_parts:
                        result_parts.append(content)
                    break
    result = '\n'.join(result_parts)
    return result[:target_len] if len(result) > target_len else result


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

    def __init__(self, level=None, near_dedup=True, json_aware=True):
        if level is None:
            level = CompressionLevel.MODERATE
        self.ic = InputCompressor(level)
        self.near_dedup = near_dedup
        self.dedup = NearDeduplicator(threshold=0.85) if near_dedup else None

    def compress(
        self,
        messages: list[dict[str, Any]],
        target_tokens: int | None = None,
        **kwargs,
    ) -> tuple[list[dict[str, Any]], dict]:
        if not messages:
            return [], {}
        ratio = kwargs.get('keep_ratio', kwargs.get('ratio', 0.5))

        # Check if content is code-heavy → skip IC, go straight to density
        full_text = "\n".join(m.get('content', '') for m in messages)
        lines = full_text.split('\n')
        is_code = _is_code_block(lines) if len(lines) > 3 else False

        if is_code:
            # Code: skip IC (it destroys code), only density compress
            compressed = list(messages)
        else:
            # Dialog: IC + density
            ic_result = self.ic.compress_messages(messages)
            compressed = ic_result[0] if isinstance(ic_result, tuple) else ic_result

        # Density compression per message
        compressed = self._density_compress_messages(compressed, ratio)

        stats = {"method": "adaptive", "keep_ratio": ratio, "skipped_ic": is_code}
        return compressed, stats

    def _density_compress_messages(self, messages, ratio):
        result = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get('content', '')
            if not content:
                result.append(msg)
                continue
            msg_target = max(1, int(len(content) * ratio))
            new_msg = dict(msg)
            new_msg['content'] = _density_compress(content, msg_target)
            result.append(new_msg)
        return result

    def get_compression_stats(self):
        return {}

    def reset_stats(self):
        pass
