"""L1: Signal/Noise Classifier v4 — Zero-cost, zero-API-call input compression.

v4 improvements over v3:
  11. Word-level fine filtering: 那么/顺便/能加/其实/就是说/反正/基本上 → noise
  12. Cross-turn verbatim instruction dedup: old-turn repeated instructions → remove
  13. History compression v2: aggressive old-turn compression + repeated instruction removal
  14. Redundant demonstrative cleanup: 那种/那种的/这个东西/那个东西 → noise
  15. Trailing punctuation normalization: consecutive ！！！/??? → single
  16. Implicit filler phrases: 也就是说/就是/就是说/怎么说呢/反正就是 → noise

Classification taxonomy:
  SIGNAL (never remove):
    - User intent/command (the verb + object, not the politeness wrapper)
    - Code snippets, error messages, stack traces
    - Key facts, numbers, names, URLs
    - Tool output data (the actual JSON data body)
    - Questions (ending with ?/？)

  NOISE (safe to remove):
    - All v3 noise types (helper-prefix, transition words, modifiers, particles)
    - Word-level fillers: 那么/顺便/能加/其实/就是说/反正/基本上/怎么说呢
    - Demonstratives: 那种/这个东西/那个东西
    - Implicit hedges: 我觉得/我认为/我想/依我看 (when not conveying opinion)
    - Cross-turn duplicate instructions (keep latest only)
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
# Noise Pattern Database (v4: word-level + v3 patterns)
# ══════════════════════════════════════════════════════════════════════════════

# Inline fillers: patterns that can appear ANYWHERE in text
INLINE_FILLERS_CN = [
    # ── v4: Word-level fine filtering ──
    r"那么[，,]?",          # "那么" as discourse marker → noise
    r"顺便(?:说一下|提一下|加一下)?",  # "顺便" variants → noise
    r"能加",                # "能加X" → "加X" (helper prefix variant)
    r"可以加",              # "可以加X" → "加X"
    r"能不能加",            # "能不能加" → "加"
    r"就是说[，,]?",        # "就是说" → noise
    r"就是[，,]?",          # standalone "就是" → noise
    r"其实[，,]?",          # "其实" → noise (hedging)
    r"反正[，,]?",          # "反正" → noise
    r"基本上[，,]?",        # "基本上" → noise
    r"怎么说呢[，,]?",      # "怎么说呢" → noise
    r"怎么说[，,]?",        # "怎么说" → noise
    r"就是就是",            # "就是就是" → noise
    r"那个[，,]?",          # demonstrative filler
    r"这个[，,]?",          # demonstrative filler
    r"那种[，,]?",          # "那种" → noise
    r"那种的[，,]?",
    r"这种[，,]?",
    r"这个东西[，,]?",
    r"那个东西[，,]?",
    r"这些东西[，,]?",
    r"那些东西[，,]?",
    r"具体来说[，,]?",      # "具体来说" → noise
    r"严格来说[，,]?",
    r"简单来说[，,]?",
    r"总的来说[，,]?",
    r"总之[，,]?",          # "总之" → noise (summarization filler)

    # ── v4: Implicit hedges ──
    r"我觉得[，,]?",        # "我觉得" → noise (hedging, not signal)
    r"我认为[，,]?",
    r"我想[，,]?",
    r"依我看[，,]?",
    r"按理说[，,]?",
    r"老实说[，,]?",
    r"说实在的[，,]?",

    # ── v3: Helper prefixes (demoted from signal → noise) ──
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
    r"好的?[，,]?",
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

    # ── v3: Trailing particles (residual) ──
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
    r"如果可以的话",
    r"要是可以的话",
    r"若可以",
    r"如果方便的话",
    r"在方便的时候",
    r"有空的话",
    r"行",
    r"嗯+",
    r"哦+",
    r"噢+",
    r"了解",
    r"明白",
    r"知道了?",
    r"收到",
    r"确实",
    r"的确",
    r"确实如此",
    r"没错",
    r"是的?",
    r"对的?",
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
    r"^HTTP/[\d.]+\s+\d+",
    r"^(Content-Type|Content-Length|Authorization|Accept|User-Agent|Referer):\s*",
    r"^(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)\s+",
    r"^\s*(status|status_code|statusCode|Status|code):\s*\d+\s*$",
    r"^\s*(request_id|requestId|request-id|trace_id|traceId|trace-id|x-request-id|x-trace|x-trace-id|x-b3-traceid):\s*[a-zA-Z0-9_-]+\s*$",
    r"^\s*(latency|duration|elapsed|response_time|took|timing):\s*[\d.]+\s*(ms|s|seconds|msec)?\s*$",
    r"^\s*(x-ratelimit|rate.?limit|retry.?after)[^\n]*$",
    r"^\s*(connection|keep-alive|transfer-encoding|cache-control|etag|x-powered-by|x-frame-options)[^\n]*$",
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

# Duplicate detection threshold
DUPLICATE_THRESHOLD = 0.8

# Cross-turn instruction dedup: similar verb-object pairs
INSTRUCTION_VERBS = re.compile(
    r'(写|创建|删除|修改|运行|执行|查询|搜索|分析|实现|优化|重构|调试|修复|安装|部署|配置|测试|比较|推荐|解释|说明|翻译|总结|生成|下载|上传|合并|检查|验证|加|添加|增加)'
)
INSTRUCTION_OBJECTS = re.compile(
    r'[\u4e00-\u9fff]{2,}'  # Chinese word sequences (2+ chars)
)


# ══════════════════════════════════════════════════════════════════════════════
# Core Classifier v4
# ══════════════════════════════════════════════════════════════════════════════

class SignalNoiseClassifier:
    """v4 classifier: word-level filtering + v3 fragment-level + inline filler stripping.

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

        all_cn = [f"(?:{p})" for p in INLINE_FILLERS_CN]
        all_en = [f"(?:{p})" for p in INLINE_FILLERS_EN]
        all_filler = all_cn + all_en
        self._master_filler = re.compile(
            "|".join(all_filler), re.IGNORECASE
        )

        # ── Precompiled patterns for _strip_redundant_quantifiers ──
        self._re_quantifier = re.compile(
            r'(写|创建|做|实现|生成|建|搭|加|添加|增加|插入|构建|构造|新建)'
            r'(?:一个|个|一下)'
        )

        # ── Precompiled patterns for _normalize_punctuation ──
        self._re_multi_excl_cn = re.compile(r'！{2,}')
        self._re_multi_excl_en = re.compile(r'!{2,}')
        self._re_multi_ques_cn = re.compile(r'？{2,}')
        self._re_multi_ques_en = re.compile(r'\?{2,}')
        self._re_multi_period_cn = re.compile(r'。{2,}')
        self._re_trailing_comma = re.compile(r'[，,]\s*$')

        # ── Precompiled patterns for _classify_fragment ──
        self._re_error_trace = re.compile(
            r'(Error|Exception|Traceback|File "|raise |assert |AttributeError|TypeError|ValueError|KeyError|IndexError|ImportError)'
        )
        self._re_url = re.compile(r'https?://\S+')
        self._re_cmd_cn = re.compile(
            r'(写|创建|删除|修改|运行|执行|查询|搜索|分析|实现|优化|重构|调试|修复|安装|部署|配置|测试|比较|推荐|解释|说明|翻译|总结|生成|下载|上传|合并|检查|验证)'
        )
        self._re_cmd_en = re.compile(
            r'\b(write|create|delete|modify|run|execute|search|analyze|implement|optimize|refactor|debug|fix|install|deploy|configure|test|compare|recommend|explain|translate|summarize|generate|download|upload|merge|check|verify|help|add|remove|update|set|get|find|show|list|open|close|enable|disable)\b',
            re.IGNORECASE
        )
        self._re_question = re.compile(r'[？?]')
        self._re_technical = re.compile(
            r'\b(Python|TypeScript|JavaScript|function|class|import|return|async|await|const|let|var|def |class |if |else|for |while )\b'
        )
        self._re_numeric = re.compile(r'^\d+[\d.,]*$')
        self._re_cn_chars = re.compile(r'[\u4e00-\u9fff]')
        self._re_excl_or_ques = re.compile(r'[？?！!]')

        # ── Precompiled pattern for _strip_fillers space cleanup ──
        self._re_multi_space = re.compile(r'\s+')

        # ── Precompiled patterns for _find_repeated_instructions ──
        self._re_instruction_clean = re.compile(r'[？?。！!，,\s]+$')

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

            if self._is_tool_noise_line(stripped):
                segments.append(Segment(
                    text=line,
                    segment_type=SegmentType.NOISE,
                    confidence=0.95,
                    reason="tool_metadata",
                ))
                i += 1
                continue

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
        """Split a line into fragments and classify each."""
        fragments = re.split(r'[,，、；;：:]\s*', line)

        result_segments = []
        for frag in fragments:
            frag = frag.strip()
            if not frag:
                continue

            stripped_frag = self._strip_fillers(frag)
            if not stripped_frag or len(stripped_frag.strip()) == 0:
                result_segments.append(Segment(
                    text=frag,
                    segment_type=SegmentType.NOISE,
                    confidence=0.85,
                    reason="filler_only",
                ))
                continue

            filler_was_removed = stripped_frag != frag

            if filler_was_removed:
                result_segments.append(Segment(
                    text=stripped_frag.strip(),
                    segment_type=SegmentType.SIGNAL,
                    confidence=0.9,
                    reason="cleaned_from_filler",
                ))
                removed_text = self._extract_removed_text(frag, stripped_frag)
                if removed_text:
                    result_segments.append(Segment(
                        text=removed_text,
                        segment_type=SegmentType.NOISE,
                        confidence=0.85,
                        reason="inline_filler",
                    ))
            else:
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

        v5: Single-pass master regex (20-50x faster than loop).
        """
        result = self._master_filler.sub("", text)

        # v3: Strip redundant quantifiers after fillers
        result = self._strip_redundant_quantifiers(result)

        # v4: Normalize punctuation
        result = self._normalize_punctuation(result)

        # Clean up extra spaces
        result = self._re_multi_space.sub(" ", result).strip()
        return result

    def _strip_redundant_quantifiers(self, text: str) -> str:
        """Strip redundant quantifiers after verbs."""
        return self._re_quantifier.sub(r'\1', text)

    def _normalize_punctuation(self, text: str) -> str:
        """v4: Normalize redundant punctuation."""
        text = self._re_multi_excl_cn.sub('！', text)
        text = self._re_multi_excl_en.sub('!', text)
        text = self._re_multi_ques_cn.sub('？', text)
        text = self._re_multi_ques_en.sub('?', text)
        text = self._re_multi_period_cn.sub('。', text)
        text = self._re_trailing_comma.sub('', text)
        return text

    def _extract_removed_text(self, original: str, cleaned: str) -> str:
        """Extract what was removed."""
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

        if stripped.startswith("```") or stripped.startswith("    "):
            return (SegmentType.SIGNAL, 1.0, "code_block")

        if self._re_error_trace.search(stripped):
            return (SegmentType.SIGNAL, 1.0, "error_trace")

        if self._re_url.search(stripped):
            return (SegmentType.SIGNAL, 0.95, "url")

        if self._re_cmd_cn.search(stripped):
            return (SegmentType.SIGNAL, 0.9, "command_cn")

        if self._re_cmd_en.search(stripped):
            return (SegmentType.SIGNAL, 0.9, "command_en")

        if self._re_question.search(stripped):
            return (SegmentType.SIGNAL, 0.9, "question")

        if self._re_technical.search(stripped):
            return (SegmentType.SIGNAL, 0.85, "technical")

        if self._re_numeric.search(stripped):
            return (SegmentType.SIGNAL, 0.8, "numeric")

        # v4: Very short Chinese fragments (< 4 chars) after filler removal
        # are often residual noise
        cn_chars = len(self._re_cn_chars.findall(stripped))
        if cn_chars <= 3 and not self._re_excl_or_ques.search(stripped):
            return (SegmentType.NOISE, 0.75, "residual_fragment")

        # Default: signal
        return (SegmentType.SIGNAL, 0.7, "default_signal")


# ══════════════════════════════════════════════════════════════════════════════
# History Compressor v2 — aggressive old-turn compression
# ══════════════════════════════════════════════════════════════════════════════

class HistoryCompressor:
    """v2: Compress old conversation history more aggressively.

    Strategy:
      - Keep last N user messages in full (N=3 by default)
      - For older turns:
        - User messages: extract only the core instruction (remove all filler/noise)
        - Assistant messages: keep code blocks + error traces + short factual summaries
        - Tool messages: strip metadata, keep only JSON data payload
      - Cross-turn: detect repeated instructions, keep only latest version
    """

    def __init__(self, keep_recent: int = 3):
        self.keep_recent = keep_recent
        self.classifier = SignalNoiseClassifier(CompressionLevel.AGGRESSIVE)

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

        # First pass: collect all user instructions for dedup detection
        user_instructions = []
        for i, msg in enumerate(old):
            if msg.get("role") == "user" and isinstance(msg.get("content", ""), str):
                user_instructions.append((i, msg["content"]))

        # Detect repeated instruction patterns across turns
        repeated_indices = self._find_repeated_instructions(user_instructions)

        for i, msg in enumerate(old):
            role = msg.get("role", "")
            content = msg.get("content", "")

            if not isinstance(content, str) or not content.strip():
                compressed_old.append(msg)
                continue

            orig_tokens = max(1, len(content) // 3)
            tokens_original += orig_tokens

            if role == "system":
                compressed_old.append(msg)
                tokens_compressed += orig_tokens
            elif role == "assistant":
                compressed_content = self._compress_assistant_reply(content)
                new_msg = msg.copy()
                new_msg["content"] = compressed_content
                compressed_old.append(new_msg)
                tokens_compressed += max(1, len(compressed_content) // 3)
            elif role == "tool":
                compressed_content = self._compress_tool_output(content)
                new_msg = msg.copy()
                new_msg["content"] = compressed_content
                compressed_old.append(new_msg)
                tokens_compressed += max(1, len(compressed_content) // 3)
            elif role == "user":
                # v4: Check if this is a repeated instruction
                if i in repeated_indices:
                    # Summarize to one-line instruction reference
                    summary = self._summarize_instruction(content)
                    new_msg = msg.copy()
                    new_msg["content"] = summary
                    compressed_old.append(new_msg)
                    tokens_compressed += max(1, len(summary) // 3)
                else:
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
            "repeated_instructions_removed": len(repeated_indices),
        }

    def _find_repeated_instructions(self, user_instructions: list[tuple[int, str]]) -> set[int]:
        """Find old user instructions that are repeated in later turns.

        Strategy: Extract verb+object pairs; if same pattern appears in
        a later old turn, mark the earlier one for summarization.
        """
        if len(user_instructions) < 2:
            return set()

        repeated = set()
        # Extract (instruction_key, index) pairs
        parsed = []
        for idx, content in user_instructions:
            key = self._extract_instruction_key(content)
            if key:
                parsed.append((idx, key))

        # For each pair, if same key appears later, mark earlier
        for i in range(len(parsed)):
            for j in range(i + 1, len(parsed)):
                if self._instructions_similar(parsed[i][1], parsed[j][1]):
                    repeated.add(parsed[i][0])

        return repeated

    def _extract_instruction_key(self, content: str) -> str:
        """Extract the core instruction verb+object from a user message."""
        # Remove all fillers first
        cleaned = self.classifier._strip_fillers(content)
        # Extract Chinese characters (the core content)
        words = re.findall(r'[\u4e00-\u9fff]+', cleaned)
        # Filter out very short words (< 2 chars)
        meaningful = [w for w in words if len(w) >= 2]
        return "".join(meaningful)

    def _instructions_similar(self, key_a: str, key_b: str) -> bool:
        """Check if two instruction keys are similar enough to dedup."""
        if not key_a or not key_b:
            return False
        # Exact match
        if key_a == key_b:
            return True
        # Substring match (one contains the other)
        if key_a in key_b or key_b in key_a:
            return True
        # Jaccard similarity on character n-grams
        def ngrams(s, n=2):
            return set(s[i:i+n] for i in range(len(s) - n + 1))
        a_ng = ngrams(key_a)
        b_ng = ngrams(key_b)
        if not a_ng or not b_ng:
            return False
        overlap = len(a_ng & b_ng) / len(a_ng | b_ng)
        return overlap > 0.6

    def _summarize_instruction(self, content: str) -> str:
        """Summarize a repeated instruction to a very short reference."""
        cleaned = self.classifier._strip_fillers(content)
        # Keep only first sentence or first 20 chars (ultra-short for max compression)
        first_sentence = re.split(r'[。！!？?\n]', cleaned)[0]
        if len(first_sentence) > 20:
            first_sentence = first_sentence[:20]
        return first_sentence.strip() or "..."

    def _compress_assistant_reply(self, content: str) -> str:
        """Compress an old assistant reply to key facts only."""
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
            if self.classifier._re_error_trace.search(stripped):
                kept.append(line)
                continue

            # Keep short factual statements
            if len(stripped) < 80 and not re.search(
                r'^(好的?|行|嗯|确实|另外|对了|补充|还有|顺便|很好|太好了|不错)',
                stripped
            ):
                kept.append(line)
                continue

            # v4: Keep lines with technical terms/keywords
            if re.search(r'(TODO|FIXME|NOTE|WARNING|HACK|关键|核心|结论)', stripped):
                kept.append(line)
                continue

        result = "\n".join(kept) if kept else content[:80] + "..."
        return result

    def _compress_tool_output(self, content: str) -> str:
        """Strip metadata from tool output, keeping only the data payload."""
        lines = content.split("\n")
        data_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

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
        segments = self.classifier.classify_text(content)

        signal_parts = []
        for seg in segments:
            if seg.segment_type in (SegmentType.SIGNAL, SegmentType.BORDERLINE):
                signal_parts.append(seg.text)

        return " ".join(signal_parts) if signal_parts else content


# ══════════════════════════════════════════════════════════════════════════════
# Tool Output Bulk Cleaner
# ══════════════════════════════════════════════════════════════════════════════

class ToolOutputCleaner:
    """Strip metadata blocks from tool/API responses."""

    METADATA_BLOCK_START = re.compile(
        r'^\s*(HTTP/|Content-Type:|Authorization:|Accept:|User-Agent:|'
        r'Referer:|X-|x-|GET |POST |PUT |DELETE |PATCH |OPTIONS |HEAD )',
        re.IGNORECASE
    )

    def clean_tool_output(self, content: str) -> str:
        """Remove metadata blocks from tool output."""
        lines = content.split("\n")
        result = []
        in_metadata_block = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_metadata_block:
                    continue
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
# Input Compressor v4 (uses all components)
# ══════════════════════════════════════════════════════════════════════════════

class InputCompressor:
    """v4: Compress chat messages using all L1 components.

    Pipeline:
      1. Clean tool outputs (strip metadata blocks)
      2. Classify each message's content (inline filler stripping)
      3. Compress old history (aggressive old-turn compression)
      4. Cross-turn dedup (repeated instructions → remove old)
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

            if not isinstance(content, str) or not content.strip():
                compressed.append(msg)
                continue

            orig_tokens = max(1, len(content) // 3)
            total_original_tokens += orig_tokens

            if role == "system" and self.level != CompressionLevel.AGGRESSIVE:
                total_compressed_tokens += orig_tokens
                compressed.append(msg)
                continue

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

            segments = self.classifier.classify_text(content)

            if not segments:
                compressed.append(msg)
                total_compressed_tokens += orig_tokens
                continue

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
        history_meta = {"compressed": False}
        if len(compressed) > self.history_compressor.keep_recent + 1:
            compressed, history_meta = self.history_compressor.compress_history(compressed)

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

        # Quality guardrail
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
