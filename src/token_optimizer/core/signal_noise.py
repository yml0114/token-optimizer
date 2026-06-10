"""L1: Signal/Noise Classifier v3 — Zero-cost, zero-API-call input compression.

v3 improvements over v2:
  6. Helper-prefix demotion: 帮我/请你/麻烦你 → noise (not signal)
  7. Transition word removal: 很好/太好了/接下来/顺便说一下 → noise
  8. Redundant modifier stripping: 完整的/功能齐全的/详细的 → noise
  9. Trailing particle cleanup: 了/吧/呢/啊/呀/嘛 → noise when residual
  10. Redundant verb-object amplification: 写一个完整的X → 写X

v2 improvements over v1:
  1. Fragment-level splitting: split on , 、 ； ; AND inline filler patterns
  2. Inline filler stripping: remove fillers from middle of sentences
  3. Tool output bulk removal: strip consecutive HTTP metadata blocks
  4. History compression: old assistant messages → key fact extraction
  5. Cross-turn dedup: repeated instructions → keep only latest

Classification taxonomy:
  SIGNAL (never remove):
    - User intent/command (the verb + object, not the politeness wrapper)
    - Code snippets, error messages, stack traces
    - Key facts, numbers, names, URLs
    - Tool output data (the actual JSON data body)
    - Questions (ending with ?/？)

  NOISE (safe to remove):
    - Helper prefixes (帮我, 请你, 麻烦你) — demoted from v2 signal
    - Filler words/phrases (请, 如果可以的话, 麻烦, etc.)
    - Politeness markers (好的谢谢, 辛苦了, 不好意思)
    - Transition words (很好, 太好了, 接下来, 顺便说一下)
    - Redundant modifiers (完整的, 功能齐全的, 详细的)
    - Residual particles (了, 吧, 呢, 啊, 呀, 嘛)
    - Tool output metadata (HTTP headers, trace_ids, status codes, latency)
    - System tags (<system_hint>, <attribution>, etc.)
    - Duplicate instructions across messages
    - Empty or near-empty messages
    - Redundant hedging (I think, I believe, maybe, perhaps)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SegmentType(Enum):
    """Classification of a text segment."""
    SIGNAL = "signal"       # Must keep
    NOISE = "noise"         # Safe to remove
    BORDERLINE = "borderline"  # Keep in safe mode, remove in moderate/aggressive


class CompressionLevel(Enum):
    """Compression aggressiveness levels."""
    SAFE = "safe"           # Only remove clear noise
    MODERATE = "moderate"   # Remove noise + borderline fluff
    AGGRESSIVE = "aggressive"  # Remove everything non-essential


@dataclass
class Segment:
    """A classified text segment."""
    text: str
    segment_type: SegmentType
    confidence: float  # 0.0-1.0
    reason: str
    start_pos: int = 0
    end_pos: int = 0

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (1 token ≈ 3 chars for mixed CJK/EN)."""
        return max(1, len(self.text) // 3)


# ══════════════════════════════════════════════════════════════════════════════
# Noise Pattern Database (v2: inline + fragment-level)
# ══════════════════════════════════════════════════════════════════════════════

# Inline fillers: patterns that can appear ANYWHERE in text (not just sentence-start)
# v3: expanded with helper-prefix demotion, transition words, redundant modifiers, trailing particles
INLINE_FILLERS_CN = [
    # ── v3: Helper prefixes (demoted from signal → noise) ──
    # "帮我写X" → "写X"; "请你创建" → "创建"; "麻烦你加" → "加"
    r"请(帮我|协助|做|实现|处理)?",
    r"麻烦(你)?",
    r"(你)?帮我",
    r"麻烦你",
    r"劳驾",

    # ── v3: Transition words / discourse markers ──
    r"很好[，,]?",
    r"太好了[，,]?",
    r"太棒了[，,]?",
    r"不错[，,]?",
    r"好的?[，,]?",  # extended to catch "好的，"
    r"接下来",
    r"然后",
    r"顺便说一下",
    r"顺便提一下",
    r"对了[，,]?",
    r"另外[，,]?",
    r"补充一下",
    r"话说回来",
    r"说到这个",
    r"这样吧",

    # ── v3: Redundant modifiers ──
    # "写一个完整的函数" → "写函数"; "创建一个功能齐全的模块" → "创建模块"
    r"一个完整的",
    r"完整的",
    r"功能齐全的",
    r"功能完善的",
    r"功能完善的",
    r"一个完善的",
    r"完善的",
    r"详细的",
    r"一个详细的",
    r"一个完整的",
    r"一个强大的",
    r"强大的",
    r"高效的",
    r"一个高效的",
    r"简单易用的",
    r"简单的",
    r"一个好的",

    # ── v3: Redundant verb-object amplification ──
    # "写一个排序算法" → "写排序算法"; "创建一个模块" → "创建模块"
    # (handled by _strip_redundant_quantifiers)

    # ── v3: Trailing particles (residual) ──
    # "辛苦了" already matched above; these catch lone particles
    r"(?:^|(?<=\s))了(?:$|(?=\s|[，,。.！!？?]))",
    r"(?:^|(?<=\s))吧(?:$|(?=\s|[，,。.！!？?]))",
    r"(?:^|(?<=\s))呢(?:$|(?=\s|[，,。.！!？?]))",
    r"(?:^|(?<=\s))啊(?:$|(?=\s|[，,。.！!？?]))",
    r"(?:^|(?<=\s))呀(?:$|(?=\s|[，,。.！!？?]))",
    r"(?:^|(?<=\s))嘛(?:$|(?=\s|[，,。.！!？?]))",

    # ── Politeness/softener (sentence-start or standalone) ──
    r"不好意思",
    r"抱歉",
    r"对不起",
    r"辛苦了?",
    # Conditional hedging (inline)
    r"如果可以的话",
    r"要是可以的话",
    r"若可以",
    r"如果方便的话",
    r"在方便的时候",
    r"有空的话",
    # Acknowledgment fillers (often standalone or at start)
    r"行",
    r"嗯+",
    r"哦+",
    r"噢+",
    r"了解",
    r"明白",
    r"知道了?",
    r"收到",
    # Confirmation fillers
    r"确实",
    r"的确",
    r"确实如此",
    r"没错",
    r"是的?",
    r"对的?",
    # Polite closings
    r"谢谢",
    r"感谢",
    r"多谢",
    r"thank you",
    r"thanks",
]

INLINE_FILLERS_EN = [
    r"please",
    r"could you",
    r"would you",
    r"can you",
    r"would it be possible",
    r"if you don't mind",
    r"when you get a chance",
    r"at your convenience",
    r"thanks",
    r"thank you",
    r"thx",
    r"cheers",
    r"ok",
    r"okay",
    r"sure",
    r"alright",
    r"got it",
    r"understood",
    r"by the way",
    r"also",
    r"additionally",
    r"furthermore",
    r"moreover",
    r"I think",
    r"I believe",
    r"I feel like",
    r"it seems like",
    r"maybe",
    r"perhaps",
    r"probably",
    r"hi",
    r"hello",
    r"hey",
    r"greetings",
]

# Tool output metadata patterns (line-level)
TOOL_OUTPUT_NOISE_LINES = [
    # HTTP lines
    r"^HTTP/[\d.]+\s+\d+",
    r"^(Content-Type|Content-Length|Authorization|Accept|User-Agent|Referer):\s*",
    r"^(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)\s+",
    # Status fields
    r"^\s*(status|status_code|statusCode|Status|code):\s*\d+\s*$",
    # Request/tracing metadata
    r"^\s*(request_id|requestId|request-id|trace_id|traceId|trace-id|x-request-id|x-trace|x-trace-id|x-b3-traceid):\s*[a-zA-Z0-9_-]+\s*$",
    # Timing
    r"^\s*(latency|duration|elapsed|response_time|took|timing):\s*[\d.]+\s*(ms|s|seconds|msec)?\s*$",
    # Rate limit headers
    r"^\s*(x-ratelimit|rate.?limit|retry.?after)[^\n]*$",
    # Connection metadata
    r"^\s*(connection|keep-alive|transfer-encoding|cache-control|etag|x-powered-by|x-frame-options)[^\n]*$",
    # Server metadata
    r"^\s*(server|date|x-request-id|x-trace-id|x-b3-traceid|x-envoy-upstream-service-time|x-ratelimit-limit|x-ratelimit-remaining|x-ratelimit-reset)[^\n]*$",
]

# System/attribution tag patterns
SYSTEM_TAG_PATTERNS = [
    r"<system_hint>.*?</system_hint>",
    r"<attribution>.*?</attribution>",
    r"<antm:[^>]*>.*?</antm:[^>]*>",
    r"<system>.*?</system>",
    r"\[SYSTEM\].*?\[/SYSTEM\]",
]

# Duplicate detection: if same instruction appears in both system prompt and user message
DUPLICATE_THRESHOLD = 0.8


# ══════════════════════════════════════════════════════════════════════════════
# Core Classifier v2
# ══════════════════════════════════════════════════════════════════════════════

class SignalNoiseClassifier:
    """v2 classifier: fragment-level splitting + inline filler stripping.

    Pipeline per text block:
      Layer 1: System tag removal (always)
      Layer 2: Line-level noise detection
      Layer 3: Fragment splitting (on ，、；;  AND inline filler boundaries)
      Layer 4: Inline filler stripping (remove fillers from within fragments)
      Layer 5: Per-fragment signal/noise classification
    """

    def __init__(self, level: CompressionLevel = CompressionLevel.MODERATE):
        self.level = level
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        self.inline_fillers_cn = [re.compile(p, re.IGNORECASE) for p in INLINE_FILLERS_CN]
        self.inline_fillers_en = [re.compile(p, re.IGNORECASE) for p in INLINE_FILLERS_EN]
        self.tool_noise_lines = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in TOOL_OUTPUT_NOISE_LINES]
        self.system_tags = [re.compile(p, re.DOTALL | re.IGNORECASE) for p in SYSTEM_TAG_PATTERNS]

        # Combined inline filler pattern for splitting
        # Build a single master pattern that matches ANY inline filler
        all_cn = [f"(?:{p})" for p in INLINE_FILLERS_CN]
        all_en = [f"(?:{p})" for p in INLINE_FILLERS_EN]
        all_filler = all_cn + all_en
        self._master_filler = re.compile(
            "|".join(all_filler), re.IGNORECASE
        )

    def classify_text(self, text: str) -> list[Segment]:
        """Classify text into signal/noise segments."""
        if not text or not text.strip():
            return []

        segments: list[Segment] = []

        # Layer 1: Strip system tags
        cleaned = text
        for pattern in self.system_tags:
            for m in reversed(list(pattern.finditer(cleaned))):
                segments.append(Segment(
                    text=m.group(),
                    segment_type=SegmentType.NOISE,
                    confidence=1.0,
                    reason="system_tag",
                    start_pos=m.start(),
                    end_pos=m.end(),
                ))
            cleaned = pattern.sub("", cleaned)

        # Layer 2: Line-level analysis
        lines = cleaned.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                segments.append(Segment(
                    text=line,
                    segment_type=SegmentType.NOISE,
                    confidence=1.0,
                    reason="empty_line",
                ))
                i += 1
                continue

            # Tool output noise (line-level)
            if self._is_tool_noise_line(stripped):
                segments.append(Segment(
                    text=line,
                    segment_type=SegmentType.NOISE,
                    confidence=0.95,
                    reason="tool_metadata",
                ))
                i += 1
                continue

            # Layer 3+4+5: Fragment-level splitting and classification
            frag_segments = self._classify_line_fragments(line)
            segments.extend(frag_segments)
            i += 1

        return segments

    def _is_tool_noise_line(self, line: str) -> bool:
        """Check if a line is tool output metadata noise."""
        for pattern in self.tool_noise_lines:
            if pattern.search(line):
                return True
        return False

    def _classify_line_fragments(self, line: str) -> list[Segment]:
        """Split a line into fragments and classify each.

        Splitting strategy:
          1. Split on Chinese/English punctuation boundaries (，、；; 、:)
          2. For each fragment, check if it's an inline filler
          3. Strip inline fillers from within fragments
          4. Classify remaining content
        """
        # Split on clause/punctuation boundaries
        # Chinese: ，、；:
        # English: , ; :
        # Also split on spaces for English
        fragments = re.split(r'[,，、；;：:]\s*', line)

        result_segments = []
        for frag in fragments:
            frag = frag.strip()
            if not frag:
                continue

            # Check if the entire fragment is a filler
            stripped_frag = self._strip_fillers(frag)
            if not stripped_frag or len(stripped_frag.strip()) == 0:
                # Entire fragment was filler
                result_segments.append(Segment(
                    text=frag,
                    segment_type=SegmentType.NOISE,
                    confidence=0.85,
                    reason="filler_only",
                ))
                continue

            # Check if stripping fillers actually removed anything
            filler_was_removed = stripped_frag != frag

            if filler_was_removed:
                # Fragment had fillers removed — classify the cleaned version as signal
                result_segments.append(Segment(
                    text=stripped_frag.strip(),
                    segment_type=SegmentType.SIGNAL,
                    confidence=0.9,
                    reason="cleaned_from_filler",
                ))
                # Also emit a noise segment for the removed fillers
                removed_text = self._extract_removed_text(frag, stripped_frag)
                if removed_text:
                    result_segments.append(Segment(
                        text=removed_text,
                        segment_type=SegmentType.NOISE,
                        confidence=0.85,
                        reason="inline_filler",
                    ))
            else:
                # No fillers found — classify as signal
                cls = self._classify_fragment(stripped_frag)
                result_segments.append(Segment(
                    text=frag,
                    segment_type=cls[0],
                    confidence=cls[1],
                    reason=cls[2],
                ))

        return result_segments

    def _strip_fillers(self, text: str) -> str:
        """Remove inline filler words from text.

        v3: Also strips redundant quantifiers (写一个X → 写X).

        Returns the cleaned text with fillers removed.
        """
        result = text
        for pattern in self.inline_fillers_cn:
            result = pattern.sub("", result)
        for pattern in self.inline_fillers_en:
            result = pattern.sub("", result)

        # v3: Strip redundant quantifiers after fillers
        # "写一个排序算法" → "写排序算法"; "创建一个模块" → "创建模块"
        # Only strip when verb (写/创建/实现/加/生成 etc.) + "一个/一下/etc."
        result = self._strip_redundant_quantifiers(result)

        # Clean up extra spaces left behind
        result = re.sub(r'\s+', ' ', result).strip()
        return result

    def _strip_redundant_quantifiers(self, text: str) -> str:
        """Strip redundant quantifiers after verbs.

        Handles patterns like:
          - 写一个排序算法 → 写排序算法
          - 做个缓存 → 做缓存
          - 实现一个函数 → 实现函数
          - 加一个原地排序版本 → 加原地排序版本
        """
        # Regex: verb + quantifier (一个/个/一下) + content
        # Only when quantifier is directly after action verb
        text = re.sub(
            r'(写|创建|做|实现|生成|建|搭|加|添加|增加|插入|构建|构造|新建)'
            r'(?:一个|一个|个|一下)',
            r'\1',
            text
        )
        # Also handle 一个/个 without verb prefix (e.g., "一个原地排序版本" → "")
        return text

    def _extract_removed_text(self, original: str, cleaned: str) -> str:
        """Extract what was removed (for debugging/reporting)."""
        # Simple approach: find characters in original that aren't in cleaned
        # This is approximate but good enough for noise tracking
        orig_chars = list(original)
        clean_chars = list(cleaned)

        removed = []
        ci = 0
        for ch in orig_chars:
            if ci < len(clean_chars) and ch == clean_chars[ci]:
                ci += 1
            else:
                removed.append(ch)

        return "".join(removed).strip()

    def _classify_fragment(self, fragment: str) -> tuple[SegmentType, float, str]:
        """Classify a single cleaned fragment."""
        stripped = fragment.strip()

        if not stripped:
            return (SegmentType.NOISE, 1.0, "empty")

        # ── High-confidence SIGNAL patterns ──

        # Code blocks
        if stripped.startswith("```") or stripped.startswith("    "):
            return (SegmentType.SIGNAL, 1.0, "code_block")

        # Error messages / stack traces
        if re.search(r'(Error|Exception|Traceback|File "|raise |assert |AttributeError|TypeError|ValueError|KeyError|IndexError|ImportError)', stripped):
            return (SegmentType.SIGNAL, 1.0, "error_trace")

        # URLs
        if re.search(r'https?://\S+', stripped):
            return (SegmentType.SIGNAL, 0.95, "url")

        # Command patterns (Chinese)
        # Note: 帮我 is demoted to noise in v3, handled by filler detection
        if re.search(r'(写|创建|删除|修改|运行|执行|查询|搜索|分析|实现|优化|重构|调试|修复|安装|部署|配置|测试|比较|推荐|解释|说明|翻译|总结|生成|下载|上传|合并|检查|验证)', stripped):
            return (SegmentType.SIGNAL, 0.9, "command_cn")

        # Command patterns (English)
        if re.search(r'\b(write|create|delete|modify|run|execute|search|analyze|implement|optimize|refactor|debug|fix|install|deploy|configure|test|compare|recommend|explain|translate|summarize|generate|download|upload|merge|check|verify|help|add|remove|update|set|get|find|show|list|open|close|enable|disable)\b', stripped, re.IGNORECASE):
            return (SegmentType.SIGNAL, 0.9, "command_en")

        # Questions
        if re.search(r'[？?]', stripped):
            return (SegmentType.SIGNAL, 0.9, "question")

        # Specific technical terms
        if re.search(r'\b(Python|TypeScript|JavaScript|function|class|import|return|async|await|const|let|var|def |class |if |else|for |while )\b', stripped):
            return (SegmentType.SIGNAL, 0.85, "technical")

        # Numbers / data points (short numeric statements)
        if re.search(r'^\d+[\d.,]*$', stripped):
            return (SegmentType.SIGNAL, 0.8, "numeric")

        # Default: signal (conservative — never delete real content)
        return (SegmentType.SIGNAL, 0.7, "default_signal")


# ══════════════════════════════════════════════════════════════════════════════
# History Compressor — compress old conversation turns
# ══════════════════════════════════════════════════════════════════════════════

class HistoryCompressor:
    """Compress old conversation history by summarizing assistant outputs.

    Strategy:
      - Keep last N user messages in full (N=3 by default)
      - For older turns: keep only key facts from assistant replies
      - Strip tool output metadata from old tool results
      - Remove empty or purely-acknowledgment messages
    """

    def __init__(self, keep_recent: int = 3):
        self.keep_recent = keep_recent

    def compress_history(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict]:
        """Compress old history messages, keeping recent ones intact.

        Returns (compressed_messages, metadata).
        """
        if len(messages) <= self.keep_recent:
            return messages, {"compressed": False, "reason": "short_history"}

        # Split: recent (keep full) vs old (compress)
        old = messages[:-self.keep_recent]
        recent = messages[-self.keep_recent:]

        compressed_old = []
        tokens_original = 0
        tokens_compressed = 0

        for msg in old:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if not isinstance(content, str) or not content.strip():
                compressed_old.append(msg)
                continue

            orig_tokens = max(1, len(content) // 3)
            tokens_original += orig_tokens

            if role == "system":
                # Keep system messages intact
                compressed_old.append(msg)
                tokens_compressed += orig_tokens
            elif role == "assistant":
                # Compress assistant replies: extract key facts
                compressed_content = self._compress_assistant_reply(content)
                new_msg = msg.copy()
                new_msg["content"] = compressed_content
                compressed_old.append(new_msg)
                tokens_compressed += max(1, len(compressed_content) // 3)
            elif role == "tool":
                # Strip tool output metadata
                compressed_content = self._compress_tool_output(content)
                new_msg = msg.copy()
                new_msg["content"] = compressed_content
                compressed_old.append(new_msg)
                tokens_compressed += max(1, len(compressed_content) // 3)
            elif role == "user":
                # Compress old user messages: keep intent, remove fillers
                compressed_content = self._compress_user_message(content)
                new_msg = msg.copy()
                new_msg["content"] = compressed_content
                compressed_old.append(new_msg)
                tokens_compressed += max(1, len(compressed_content) // 3)

        result = compressed_old + recent
        savings = tokens_original - tokens_compressed
        ratio = tokens_compressed / max(1, tokens_original)

        return result, {
            "compressed": True,
            "old_turns_compressed": len(old),
            "recent_kept": self.keep_recent,
            "original_tokens_est": tokens_original,
            "compressed_tokens_est": tokens_compressed,
            "savings_tokens": savings,
            "savings_pct": round((1 - ratio) * 100, 1),
        }

    def _compress_assistant_reply(self, content: str) -> str:
        """Compress an old assistant reply to key facts only.

        Strategy: Keep code blocks, error messages, and factual statements.
        Remove verbose explanations, acknowledgments, and filler.
        """
        lines = content.split("\n")
        kept = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Always keep code blocks
            if stripped.startswith("```") or stripped.startswith("    "):
                kept.append(line)
                continue

            # Keep error messages
            if re.search(r'(Error|Exception|Traceback|File "|raise )', stripped):
                kept.append(line)
                continue

            # Keep short factual statements (likely key points)
            if len(stripped) < 100 and not re.search(
                r'^(好的?|行|嗯|确实|另外|对了|补充|还有|顺便)',
                stripped
            ):
                kept.append(line)
                continue

            # Remove long explanations and verbose content
            # (these are the "filler" in assistant responses)

        result = "\n".join(kept) if kept else content[:100] + "..."
        return result

    def _compress_tool_output(self, content: str) -> str:
        """Strip metadata from tool output, keeping only the data payload."""
        lines = content.split("\n")
        data_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Remove HTTP headers, metadata, trace info
            is_metadata = False
            for pattern in TOOL_OUTPUT_NOISE_LINES:
                if re.match(pattern, stripped, re.IGNORECASE):
                    is_metadata = True
                    break

            if not is_metadata:
                data_lines.append(line)

        return "\n".join(data_lines) if data_lines else content

    def _compress_user_message(self, content: str) -> str:
        """Compress old user messages: remove fillers, keep intent."""
        classifier = SignalNoiseClassifier(CompressionLevel.AGGRESSIVE)
        segments = classifier.classify_text(content)

        signal_parts = []
        for seg in segments:
            if seg.segment_type in (SegmentType.SIGNAL, SegmentType.BORDERLINE):
                signal_parts.append(seg.text)

        return " ".join(signal_parts) if signal_parts else content


# ══════════════════════════════════════════════════════════════════════════════
# Tool Output Bulk Cleaner
# ══════════════════════════════════════════════════════════════════════════════

class ToolOutputCleaner:
    """Strip metadata blocks from tool/API responses.

    Handles:
      - HTTP header blocks (consecutive metadata lines)
      - JSON wrapper metadata (status, trace_id, latency)
      - Rate limit headers
      - Connection metadata
      - Server metadata
    """

    # Patterns for blocks of consecutive metadata
    METADATA_BLOCK_START = re.compile(
        r'^\s*(HTTP/|Content-Type:|Authorization:|Accept:|User-Agent:|'
        r'Referer:|X-|x-|GET |POST |PUT |DELETE |PATCH |OPTIONS |HEAD )',
        re.IGNORECASE
    )

    def clean_tool_output(self, content: str) -> str:
        """Remove metadata blocks from tool output.

        Strategy: Find consecutive metadata lines and remove the whole block.
        Keep JSON data payloads and actual response content.
        """
        lines = content.split("\n")
        result = []
        in_metadata_block = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_metadata_block:
                    continue  # Skip empty lines within metadata blocks
                result.append(line)
                continue

            is_metadata = self._is_metadata_line(stripped)

            if is_metadata:
                in_metadata_block = True
                continue
            else:
                in_metadata_block = False
                result.append(line)

        return "\n".join(result)

    def _is_metadata_line(self, line: str) -> bool:
        """Check if a line is metadata."""
        for pattern in TOOL_OUTPUT_NOISE_LINES:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Input Compressor v2 (uses all components)
# ══════════════════════════════════════════════════════════════════════════════

class InputCompressor:
    """v2: Compress chat messages using all L1 components.

    Pipeline:
      1. Clean tool outputs (strip metadata blocks)
      2. Classify each message's content (inline filler stripping)
      3. Compress old history (summarize old turns)
      4. Deduplicate instructions across messages
      5. Rebuild clean message list

    Quality guardrails:
      - Never compress below 30% of original tokens
      - Never remove ALL content from any single message
      - Always preserve code blocks, errors, URLs, questions
    """

    MIN_COMPRESSION_RATIO = 0.30

    def __init__(self, level: CompressionLevel = CompressionLevel.MODERATE):
        self.level = level
        self.classifier = SignalNoiseClassifier(level)
        self.tool_cleaner = ToolOutputCleaner()
        self.history_compressor = HistoryCompressor(keep_recent=3)

    def compress_messages(
        self, messages: list[dict[str, Any]],
        system_text: str = "",
    ) -> tuple[list[dict[str, Any]], dict]:
        """Compress a list of chat messages."""
        total_original_tokens = 0
        total_compressed_tokens = 0
        total_noise_removed = 0
        messages_changed = 0

        compressed = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Non-text content: pass through
            if not isinstance(content, str) or not content.strip():
                compressed.append(msg)
                continue

            # Count original tokens
            orig_tokens = max(1, len(content) // 3)
            total_original_tokens += orig_tokens

            # System messages: only compress in AGGRESSIVE mode
            if role == "system" and self.level != CompressionLevel.AGGRESSIVE:
                total_compressed_tokens += orig_tokens
                compressed.append(msg)
                continue

            # Tool messages: clean metadata
            if role == "tool":
                cleaned = self.tool_cleaner.clean_tool_output(content)
                comp_tokens = max(1, len(cleaned) // 3)
                total_compressed_tokens += comp_tokens
                total_noise_removed += max(0, orig_tokens - comp_tokens)
                if cleaned != content:
                    messages_changed += 1
                new_msg = msg.copy()
                new_msg["content"] = cleaned
                compressed.append(new_msg)
                continue

            # User/Assistant messages: classify and strip fillers
            segments = self.classifier.classify_text(content)

            if not segments:
                compressed.append(msg)
                total_compressed_tokens += orig_tokens
                continue

            # Filter segments based on compression level
            kept_segments = []
            for seg in segments:
                if seg.segment_type == SegmentType.SIGNAL:
                    kept_segments.append(seg)
                elif seg.segment_type == SegmentType.NOISE:
                    total_noise_removed += seg.token_estimate
                elif seg.segment_type == SegmentType.BORDERLINE:
                    if self.level in (CompressionLevel.MODERATE, CompressionLevel.AGGRESSIVE):
                        total_noise_removed += seg.token_estimate
                    else:
                        kept_segments.append(seg)

            if kept_segments:
                compressed_content = " ".join(s.text for s in kept_segments)
                comp_tokens = sum(s.token_estimate for s in kept_segments)
            else:
                compressed_content = content
                comp_tokens = orig_tokens

            total_compressed_tokens += comp_tokens
            if compressed_content != content:
                messages_changed += 1

            new_msg = msg.copy()
            new_msg["content"] = compressed_content
            compressed.append(new_msg)

        # ── History compression: compress old turns ──
        if len(compressed) > self.history_compressor.keep_recent + 1:
            compressed, history_meta = self.history_compressor.compress_history(compressed)
        else:
            history_meta = {"compressed": False}

        # ── Cross-message deduplication ──
        dedup_removed = 0
        if system_text:
            compressed, dedup_removed = self._deduplicate_across_messages(
                compressed, system_text
            )

        # Recalculate after history compression
        total_compressed_tokens = sum(
            max(1, len(m.get("content", "")) // 3)
            for m in compressed
            if isinstance(m.get("content", ""), str)
        )

        # ── Empty messages ──
        if not messages:
            return [], {
                "compressed": True,
                "level": self.level.value,
                "original_tokens_est": 0,
                "compressed_tokens_est": 0,
                "compression_ratio": 1.0,
                "savings_pct": 0.0,
                "noise_removed_tokens": 0,
                "messages_processed": 0,
                "messages_compressed": 0,
            }

        # Quality guardrail (skip for tool-only messages — metadata is always safe to strip)
        has_non_tool_content = any(
            m.get("role") in ("user", "assistant") and isinstance(m.get("content", ""), str) and m["content"].strip()
            for m in messages
        )
        if has_non_tool_content:
            actual_ratio = total_compressed_tokens / max(1, total_original_tokens)
            if actual_ratio < self.MIN_COMPRESSION_RATIO:
                return messages, {
                    "compressed": False,
                    "reason": "would_exceed_min_ratio",
                    "original_tokens_est": total_original_tokens,
                    "compressed_tokens_est": total_original_tokens,
                }
        else:
            # Tool-only messages: no guardrail, just calculate ratio
            actual_ratio = total_compressed_tokens / max(1, total_original_tokens)

        metadata = {
            "compressed": True,
            "level": self.level.value,
            "original_tokens_est": total_original_tokens,
            "compressed_tokens_est": total_compressed_tokens,
            "compression_ratio": round(actual_ratio, 3),
            "savings_pct": round((1 - actual_ratio) * 100, 1),
            "noise_removed_tokens": total_noise_removed,
            "history_compression": history_meta,
            "messages_processed": len(messages),
            "messages_compressed": messages_changed,
        }

        return compressed, metadata

    def _deduplicate_across_messages(
        self,
        messages: list[dict[str, Any]],
        system_text: str,
    ) -> tuple[list[dict[str, Any]], int]:
        """Remove user messages that merely echo system instructions."""
        tokens_removed = 0
        system_phrases = self._extract_key_phrases(system_text)

        deduplicated = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role in ("user", "assistant") and isinstance(content, str):
                msg_phrases = self._extract_key_phrases(content)
                overlap = self._phrase_overlap(system_phrases, msg_phrases)

                if overlap > DUPLICATE_THRESHOLD and len(content) < 200:
                    tokens_removed += max(1, len(content) // 3)
                    continue

            deduplicated.append(msg)

        return deduplicated, tokens_removed

    def _extract_key_phrases(self, text: str) -> set[str]:
        words = re.findall(r'[\w\u4e00-\u9fff]+', text.lower())
        return set(words)

    def _phrase_overlap(self, set_a: set[str], set_b: set[str]) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)
