"""
核心思想：不同内容类型用不同压缩策略。
- JSON/API → JSON-aware + 较高 ratio 保结构与字段
- 代码 → 跳过 IC，代码块整块保护，避免数值/常量被去噪删掉
- 短事实/规格 → fact-preserving extract，优先保留原子事实（数值、单位、KV、专有名词）
- 普通对话 → IC 去噪 + 密度压缩

Phase 5b/5c:
- 修复 density_compress 的 id() 去重失效问题，改为内容集合
- 代码跳过 InputCompressor
- 新增 fact-preserving path，解决短对话/技术规格关键数值丢失
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


# ── 事实密度检测/抽取 ──────────────────────────────────────────────────────

_FACT_UNIT_RE = _re.compile(
    r'\b\d[\d,.]*\s*(?:%|％|ms|s|sec|req/sec|GB|MB|KB|TB|vCPU|CPU|RAM|SSD|NVMe|K|M|mo|month|day|week|DAGs?|connections?|instances?|nodes?|brokers?)\b',
    _re.IGNORECASE,
)
_NUMBER_RE = _re.compile(r'[$¥€£]?\d[\d,.]*(?:[KkMm]|%|％)?')
_KV_RE = _re.compile(r'[A-Za-z_][\w\s/-]{0,40}[:：=→]\s*\S')
_IDENTIFIER_RE = _re.compile(r'\b[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b')
_COMPARE_RE = _re.compile(r'(?:<=|>=|<|>|===|!==|==|!=)')
_BULLET_RE = _re.compile(r'^\s*[-•*]\s+\S')


def _is_fact_dense_text(text: str) -> bool:
    """High-density fact/spec/short QA content should not go through IC+generic density twice."""
    if not text or len(text.strip()) < 20:
        return False
    numbers = len(_NUMBER_RE.findall(text))
    units = len(_FACT_UNIT_RE.findall(text))
    kvs = len(_KV_RE.findall(text))
    bullets = len(_re.findall(r'^\s*[-•*]\s+\S', text, _re.MULTILINE))
    arrows = text.count('→') + text.count('->')

    # Short weather / factual answers: few sentences, several values.
    if len(text) < 500 and numbers >= 3:
        return True

    # Specs/config/capacity plans: bullets + values/units/KV.
    fact_score = numbers * 1.0 + units * 2.5 + kvs * 2.0 + bullets * 1.5 + arrows * 2.0
    return fact_score > max(len(text) * 0.045, 6)


def _split_fact_chunks(text: str) -> List[str]:
    chunks: List[str] = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Bullet/spec lines are atomic facts; keep them intact.
        if _BULLET_RE.match(line) or _KV_RE.search(line):
            chunks.append(line)
            continue
        # Otherwise split sentence-like content, but keep numeric clauses together.
        parts = _re.split(r'(?<=[.!?。！？])\s+', line)
        for part in parts:
            part = part.strip()
            if part:
                chunks.append(part)
    return chunks


def _fact_score(chunk: str) -> float:
    text = chunk.strip()
    if not text:
        return 0.0
    score = 0.0
    score += len(_FACT_UNIT_RE.findall(text)) * 10.0
    score += len(_NUMBER_RE.findall(text)) * 5.0
    score += len(_KV_RE.findall(text)) * 8.0
    score += len(_IDENTIFIER_RE.findall(text)) * 7.0
    score += len(_COMPARE_RE.findall(text)) * 4.0
    score += len(_re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', text)) * 2.0
    score += len(_re.findall(r'\b[A-Z]{2,}\b', text)) * 2.0

    # Domain-critical hints often appear in benchmark/real specs.
    critical_words = [
        'throughput', 'latency', 'p99', 'connection', 'pool', 'cost', 'total',
        'budget', 'total budget', 'owner', 'due', 'status', 'from', 'to', 'migrate', 'risk',
        'delay', 'delayed', 'eta', 'compliance', 'review', 'critical path',
        'combined', 'completion', 'weighted average', 'roughly', 'invoice', 'invoice_id',
        'customer', 'region', 'alert_threshold', 'estimated_cost',
        'temperature', 'humidity', 'rain', 'wind', 'discount', 'validation',
    ]
    low = text.lower()
    score += sum(4.0 for w in critical_words if w in low)

    # Shorter factual chunks are more efficient.
    return score / max(len(text) ** 0.25, 1)


def _compact_fact_chunk(chunk: str) -> str:
    """Make facts denser without deleting values."""
    s = chunk.strip()
    s = _re.sub(r'^\s*[-•*]\s+', '', s)
    s = s.replace(' instances', ' inst').replace(' instance', ' inst')
    s = s.replace(' servers', ' srv').replace(' server', ' srv')
    s = s.replace(' monthly cost', ' cost')
    s = s.replace('Total monthly cost', 'Cost')
    s = s.replace('Peak throughput', 'Throughput')
    s = s.replace('latency target', 'latency')
    s = s.replace('DB connection pool', 'DB pool')
    s = s.replace('per web server', '/web')
    s = _re.sub(r'\s+', ' ', s)
    return s


def _fact_preserving_compress(text: str, target_len: int) -> str:
    if target_len >= len(text):
        return text

    chunks = _split_fact_chunks(text)
    if not chunks:
        return _density_compress(text, target_len)

    compacted = [_compact_fact_chunk(c) for c in chunks]
    scored = [(i, c, _fact_score(c)) for i, c in enumerate(compacted)]

    # First keep fact-bearing chunks in score order.
    kept_idx = set()
    total = 0
    for i, c, score in sorted(scored, key=lambda x: x[2], reverse=True):
        if score <= 0:
            continue
        add = len(c) + (1 if total else 0)
        if total + add <= target_len:
            kept_idx.add(i)
            total += add

    # If budget remains, fill by original order for coherence.
    for i, c, score in scored:
        if i in kept_idx:
            continue
        add = len(c) + (1 if total else 0)
        if total + add <= target_len:
            kept_idx.add(i)
            total += add

    if not kept_idx:
        return _density_compress(text, target_len)

    result = '\n'.join(compacted[i] for i in sorted(kept_idx))
    return result[:target_len] if len(result) > target_len else result


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
    # Boost fact-bearing spec lines.
    score += len(_FACT_UNIT_RE.findall(text)) * 5.0
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
        # Never character-truncate dense facts/code: partial tokens like "compli"
        # destroy exactly the facts QA needs. Allow slight budget overshoot.
        if _is_fact_dense_text(text) or _fact_score(text) > 4.0 or _looks_like_code(text):
            return _compact_fact_chunk(text)
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
    emitted_code_blocks = set()
    for chunk in raw_chunks:
        if chunk in kept_content:
            result_parts.append(chunk)
        else:
            for content, _, ctype in scored_chunks:
                if ctype == 'code' and content in kept_content and chunk in content:
                    if content not in emitted_code_blocks:
                        result_parts.append(content)
                        emitted_code_blocks.add(content)
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

        full_text = "\n".join(m.get('content', '') for m in messages)
        lines = full_text.split('\n')
        is_code = _is_code_block(lines) if len(lines) > 3 else False
        is_fact_dense = _is_fact_dense_text(full_text)

        # Short fact-dense content has almost no safe redundancy.
        # If the policy would only save a few tokens, no-op is safer than dropping
        # conditions such as `balance < estimated_cost` or derived totals like `$4,500`.
        if (
            not is_code
            and len(full_text) < 1200
            and ratio >= 0.90
            and (
                is_fact_dense
                or len(_NUMBER_RE.findall(full_text)) >= 6
                or len(_IDENTIFIER_RE.findall(full_text)) >= 1
            )
        ):
            stats = {
                "method": "adaptive",
                "keep_ratio": 1.0,
                "skipped_ic": True,
                "code_mode": False,
                "fact_mode": True,
                "no_op_short_fact": True,
            }
            return list(messages), stats

        # Code/fact-dense content: skip IC; IC may delete exactly the values we need.
        if is_code or is_fact_dense:
            compressed = list(messages)
            skipped_ic = True
        else:
            ic_result = self.ic.compress_messages(messages)
            compressed = ic_result[0] if isinstance(ic_result, tuple) else ic_result
            skipped_ic = False

        compressed = self._density_compress_messages(compressed, ratio, fact_mode=is_fact_dense and not is_code)

        stats = {
            "method": "adaptive",
            "keep_ratio": ratio,
            "skipped_ic": skipped_ic,
            "code_mode": is_code,
            "fact_mode": is_fact_dense and not is_code,
        }
        return compressed, stats

    def _density_compress_messages(self, messages, ratio, fact_mode=False):
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
            if fact_mode or _is_fact_dense_text(content):
                new_msg['content'] = _fact_preserving_compress(content, msg_target)
            else:
                new_msg['content'] = _density_compress(content, msg_target)
            result.append(new_msg)
        return result

    def get_compression_stats(self):
        return {}

    def reset_stats(self):
        pass
