"""L0: Prefix Structure Optimizer — Zero-cost, zero-quality-impact reorder.

This is the highest-ROI layer. It reorders prompt messages so that static content
(system prompt, tool definitions) is at the front, and dynamic content (user input,
timestamps) is at the end. This maximizes API cache hit rates.

DeepSeek caches from token 0 forward. If your prefix changes every request,
cache never hits. This module ensures it always hits.

v2 Enhancement (Headroom CacheAligner integration):
  DynamicContentDetector identifies and extracts time-varying content from
  system prompts (dates, UUIDs, session IDs, timestamps). These are moved
  to a _dynamic_tail region so the static prefix stays stable across
  consecutive requests, dramatically improving KV cache hit rate.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# DynamicContentDetector (Headroom CacheAligner)
# ══════════════════════════════════════════════════════════════════════════════

class DynamicContentDetector:
    """Detect and extract time-varying content from system prompts.

    In many production setups, system prompts contain session-specific or
    time-varying fields like current dates, UUIDs, session IDs, and
    timestamps. Each change in these fields invalidates the KV cache for
    the entire prefix.

    This detector extracts such dynamic content from the system prompt and
    moves it to a `_dynamic_tail` region appended after the system message.
    The main system prompt becomes stable across requests.

    Usage:
        detector = DynamicContentDetector()
        stable_text, tail = detector.extract_dynamic("Today is 2026-06-11...")
        # stable_text: "Today is {DATE}..."
        # tail: {"date_0": "2026-06-11", ...}

        # In subsequent requests, inject tail back:
        full_text = detector.inject_tail(stable_text, tail)
    """

    # ── Date patterns (ISO, US, EU, textual) ──
    _DATE_PATTERNS: list[tuple[str, str, str]] = [
        # ISO format: 2026-06-11
        (r'\b\d{4}-\d{2}-\d{2}\b', '{DATE}', 'iso_date'),
        # Slash format: 2026/06/11
        (r'\b\d{4}/\d{2}/\d{2}\b', '{DATE}', 'slash_date'),
        # US format: 06/11/2026
        (r'\b\d{2}/\d{2}/\d{4}\b', '{DATE}', 'us_date'),
        # Textual: Jun 11, 2026 / June 11, 2026
        (r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
         '{DATE}', 'textual_date'),
        # Chinese: 2026年06月11日
        (r'\b\d{4}年\d{1,2}月\d{1,2}日?\b', '{DATE}', 'cn_date'),
        # Day of week: Monday, 2026-06-11 / 周三 2026-06-11
        (r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,?\s+\d{4}-\d{2}-\d{2}\b',
         '{DATE}', 'weekday_date'),
        (r'周[一二三四五六日天]\s*\d{4}-\d{2}-\d{2}', '{DATE}', 'cn_weekday_date'),
    ]

    # ── UUID pattern ──
    _UUID_PATTERN = (
        r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'
    )

    # ── Session ID patterns ──
    _SESSION_PATTERNS = [
        r'\bsession[_-][A-Za-z0-9_-]{8,}',
        r'\bsid[_-][A-Za-z0-9_-]{8,}',
        r'\breq[_-][A-Za-z0-9_-]{8,}',
        r'\btrace[_-][A-Za-z0-9_-]{8,}',
        r'\bconv[_-][A-Za-z0-9_-]{8,}',
        r'\bconv_id[_-][A-Za-z0-9_-]{6,}',
        r'\brequest[_-]id[_-][A-Za-z0-9_-]{6,}',
    ]

    # ── Timestamp patterns ──
    _TIMESTAMP_PATTERNS = [
        # Unix timestamp (10-13 digits, ms or s precision)
        r'\b\d{10}(?:\.\d{1,6})?\b',
        r'\b\d{13}\b',
        # ISO 8601: 2026-06-11T03:56:00Z or with timezone
        r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b',
    ]

    def __init__(self) -> None:
        self._compiled_date = [
            (re.compile(p), repl, name) for p, repl, name in self._DATE_PATTERNS
        ]
        self._compiled_uuid = re.compile(self._UUID_PATTERN)
        self._compiled_session = [re.compile(p) for p in self._SESSION_PATTERNS]
        self._compiled_ts = [re.compile(p) for p in self._TIMESTAMP_PATTERNS]

    def detect_dynamic_fields(self, text: str) -> dict[str, str]:
        """Detect all dynamic fields in the given text.

        Returns:
            Dict mapping field_name → original_value for each detected field.
        """
        fields: dict[str, str] = {}
        counter: dict[str, int] = {}

        # Dates
        for pattern, placeholder, name in self._compiled_date:
            for match in pattern.finditer(text):
                key = f"{name}_{counter.get(name, 0)}"
                counter[name] = counter.get(name, 0) + 1
                fields[key] = match.group()

        # UUIDs
        for match in self._compiled_uuid.finditer(text):
            key = f"uuid_{counter.get('uuid', 0)}"
            counter['uuid'] = counter.get('uuid', 0) + 1
            fields[key] = match.group()

        # Session IDs
        for pattern in self._compiled_session:
            for match in pattern.finditer(text):
                key = f"session_{counter.get('session', 0)}"
                counter['session'] = counter.get('session', 0) + 1
                fields[key] = match.group()

        # Timestamps (skip if already matched as date)
        for pattern in self._compiled_ts:
            for match in pattern.finditer(text):
                value = match.group()
                # Skip if this is already captured as a date
                if any(v == value for v in fields.values()):
                    continue
                key = f"timestamp_{counter.get('timestamp', 0)}"
                counter['timestamp'] = counter.get('timestamp', 0) + 1
                fields[key] = value

        return fields

    def extract_dynamic(
        self, text: str
    ) -> tuple[str, dict[str, str]]:
        """Extract dynamic content from text, returning stable template + tail.

        Replaces detected dynamic fields with placeholders in the text and
        stores original values in the tail dict.

        Args:
            text: Original system prompt text.

        Returns:
            Tuple of (stable_text_with_placeholders, dynamic_tail_dict).
        """
        fields = self.detect_dynamic_fields(text)
        if not fields:
            return text, {}

        stable = text

        # Replace dates
        for pattern, placeholder, _ in self._compiled_date:
            stable = pattern.sub(placeholder, stable)

        # Replace UUIDs
        stable = self._compiled_uuid.sub('{UUID}', stable)

        # Replace session IDs
        for pattern in self._compiled_session:
            stable = pattern.sub('{SESSION_ID}', stable)

        # Replace timestamps
        for pattern in self._compiled_ts:
            stable = pattern.sub('{TIMESTAMP}', stable)

        return stable, fields

    def inject_tail(self, stable_text: str, tail: dict[str, str]) -> str:
        """Re-inject dynamic fields from tail back into stable text.

        This reconstructs the original system prompt with current dynamic values.

        Args:
            stable_text: Text with placeholders.
            tail: Dynamic field values from extract_dynamic.

        Returns:
            Fully populated text with dynamic values restored.
        """
        if not tail:
            return stable_text

        result = stable_text

        # Group replacements by type
        date_idx = 0
        uuid_idx = 0
        session_idx = 0
        ts_idx = 0

        for key, value in tail.items():
            if key.startswith('iso_date') or key.startswith('slash_date') or \
               key.startswith('us_date') or key.startswith('textual_date') or \
               key.startswith('cn_date') or key.startswith('weekday_date'):
                # Replace first occurrence of {DATE}
                result = result.replace('{DATE}', value, 1)
            elif key.startswith('uuid'):
                result = result.replace('{UUID}', value, 1)
            elif key.startswith('session'):
                result = result.replace('{SESSION_ID}', value, 1)
            elif key.startswith('timestamp'):
                result = result.replace('{TIMESTAMP}', value, 1)

        return result

    def extract_and_move_to_tail(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process messages: extract dynamic fields from system prompts and
        append them as a _dynamic_tail message at the end.

        This is the main integration point for the reorder pipeline.

        Args:
            messages: List of message dicts.

        Returns:
            Modified messages with dynamic content moved to tail.
        """
        if not messages:
            return messages

        result = []
        dynamic_tails: list[dict[str, str]] = []

        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                stable, tail = self.extract_dynamic(msg["content"])
                if tail:
                    # Replace system message content with stable version
                    new_msg = {**msg, "content": stable}
                    result.append(new_msg)
                    dynamic_tails.append(tail)
                else:
                    result.append(msg)
            else:
                result.append(msg)

        # Append combined dynamic tails as a special message
        if dynamic_tails:
            combined_tail: dict[str, str] = {}
            for tail in dynamic_tails:
                combined_tail.update(tail)

            # Create a structured tail message
            tail_parts = []
            for key, value in combined_tail.items():
                tail_parts.append(f"{key}={value}")

            result.append({
                "role": "system",
                "content": "[DYNAMIC_CONTEXT]\n" + "\n".join(tail_parts) + "\n[/DYNAMIC_CONTEXT]",
                "_dynamic_tail": True,
            })

        return result

    def get_prefix_stability_score(
        self,
        messages_a: list[dict[str, Any]],
        messages_b: list[dict[str, Any]],
    ) -> float:
        """Compute prefix stability score between two request message sets.

        Higher score = more stable prefix = better cache hit rate.

        Returns:
            Float between 0.0 and 1.0 representing prefix similarity.
        """
        def _extract_prefix(msgs: list[dict[str, Any]]) -> str:
            """Extract the static prefix portion (system + tools, excluding last user)."""
            parts = []
            for msg in msgs[:-1] if len(msgs) > 1 else msgs:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("system", "tool_def") and isinstance(content, str):
                    # Remove dynamic fields for comparison
                    stable, _ = self.extract_dynamic(content)
                    parts.append(f"{role}:{stable}")
            return "\n".join(parts)

        prefix_a = _extract_prefix(messages_a)
        prefix_b = _extract_prefix(messages_b)

        if not prefix_a and not prefix_b:
            return 1.0
        if not prefix_a or not prefix_b:
            return 0.0

        # Simple Jaccard-like similarity on character n-grams
        def _ngrams(text: str, n: int = 4) -> set[str]:
            return {text[i:i+n] for i in range(len(text) - n + 1)}

        grams_a = _ngrams(prefix_a)
        grams_b = _ngrams(prefix_b)

        if not grams_a and not grams_b:
            return 1.0

        intersection = grams_a & grams_b
        union = grams_a | grams_b
        return len(intersection) / len(union) if union else 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Original reorder functions (preserved, with enhanced strip_dynamic_fields)
# ══════════════════════════════════════════════════════════════════════════════

# Singleton detector for reuse across calls
_detector = DynamicContentDetector()


def reorder_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    enable_dynamic_extraction: bool = True,
) -> tuple[list[dict[str, Any]], dict]:
    """Reorder messages for maximum cache hit rate.

    New order:
        1. system prompt (static, cacheable)
        2. tool definitions (static, cacheable)
        3. conversation memory / summary (quasi-static)
        4. history messages (growing, tail-cacheable)
        5. current user input (dynamic, not cacheable)
        6. _dynamic_tail (dynamic fields extracted from system prompt)

    Args:
        messages: Original message list.
        tools: Optional tool definitions.
        enable_dynamic_extraction: If True, use DynamicContentDetector to
            extract and move dynamic content to tail.

    Returns:
        (reordered_messages, metadata) where metadata has hash info
    """
    if not messages:
        return messages, {"prefix_hash": "", "prefix_tokens_est": 0}

    # Step 0: Extract dynamic content from system prompts
    if enable_dynamic_extraction:
        messages = _detector.extract_and_move_to_tail(messages)

    # Separate message types
    system_msgs: list[dict[str, Any]] = []
    user_msgs: list[dict[str, Any]] = []
    assistant_msgs: list[dict[str, Any]] = []
    tool_msgs: list[dict[str, Any]] = []
    dynamic_tail_msgs: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        if msg.get("_dynamic_tail"):
            dynamic_tail_msgs.append(msg)
        elif role == "system":
            system_msgs.append(msg)
        elif role == "user":
            user_msgs.append(msg)
        elif role == "assistant":
            assistant_msgs.append(msg)
        elif role == "tool":
            tool_msgs.append(msg)

    # Build reordered message list
    reordered: list[dict[str, Any]] = []

    # 1. System prompt(s) — always first, always static
    reordered.extend(system_msgs)

    # 2. Tool definitions — after system, before history
    if tools:
        # Sort tools by name for deterministic ordering
        sorted_tools = sorted(tools, key=lambda t: t.get("function", {}).get("name", ""))
        for tool in sorted_tools:
            reordered.append({"role": "tool_def", "tool": tool})

    # 3. History: interleave assistant + user (keep temporal order)
    #    Move the last user message to the very end
    if user_msgs and assistant_msgs:
        current_user = user_msgs[-1] if user_msgs else None

        # Interleave by timestamp (assume original order)
        for msg in messages:
            if msg.get("role") in ("user", "assistant") and msg is not current_user:
                if not msg.get("_dynamic_tail"):
                    reordered.append(msg)

        # Tool messages go near their corresponding assistant messages
        reordered.extend(tool_msgs)

        # 5. Current user input — always last
        if current_user:
            reordered.append(current_user)
    elif user_msgs:
        reordered.extend(user_msgs)

    # 6. Dynamic tail — appended at the very end (after last user message)
    #    This keeps the main prefix stable while preserving dynamic context
    reordered.extend(dynamic_tail_msgs)

    # Compute prefix hash (everything except the last user message and dynamic tail)
    # The prefix should be the stable part only
    prefix_msgs = [m for m in reordered if not m.get("_dynamic_tail")]
    if len(prefix_msgs) > 1:
        prefix_msgs_for_hash = prefix_msgs[:-1]
    else:
        prefix_msgs_for_hash = prefix_msgs

    prefix_content = _serialize_for_hash(prefix_msgs_for_hash)
    prefix_hash = hashlib.sha256(prefix_content.encode()).hexdigest()[:16]

    # Estimate prefix token count (rough: 1 token ≈ 4 chars for English, ~2 for CJK)
    prefix_text = _extract_text(prefix_msgs[:-1] if prefix_msgs else [])
    prefix_tokens_est = max(1, len(prefix_text) // 3)  # rough average

    # Count dynamic fields that were extracted
    dynamic_fields_count = sum(
        1 for msg in dynamic_tail_msgs
        for _ in (msg.get("content", "").split("\n"))
        if "=" in _
    )

    metadata = {
        "prefix_hash": prefix_hash,
        "prefix_tokens_est": prefix_tokens_est,
        "original_order": [m.get("role", "?") for m in messages],
        "reordered_order": [m.get("role", "?") for m in reordered],
        "did_reorder": reordered != messages,
        "dynamic_fields_extracted": dynamic_fields_count,
        "has_dynamic_tail": len(dynamic_tail_msgs) > 0,
    }

    return reordered, metadata


def compute_prefix_hash(messages: list[dict[str, Any]]) -> str:
    """Compute a stable hash of the message prefix (everything except last user msg).

    This hash is used to track whether the prefix has changed between requests.
    If hash == previous hash, cache will hit.
    """
    if len(messages) <= 1:
        content = _serialize_for_hash(messages)
    else:
        content = _serialize_for_hash(messages[:-1])
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def strip_dynamic_fields(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove fields that change between requests but don't affect meaning.

    Enhanced v2: Now uses DynamicContentDetector for more comprehensive
    detection of dates, UUIDs, session IDs, and timestamps in message content.

    E.g., timestamps, request IDs, attribution blocks.
    """
    STRIPPED_KEYS = {"timestamp", "ts", "request_id", "trace_id", "attribution"}

    cleaned = []
    for msg in messages:
        cleaned_msg = {k: v for k, v in msg.items() if k not in STRIPPED_KEYS}
        # Remove attribution blocks from content (common in Claude Code / MiMo)
        if isinstance(cleaned_msg.get("content"), str):
            cleaned_msg["content"] = _strip_attribution_block(cleaned_msg["content"])
            # v2: Use DynamicContentDetector to strip dynamic content from ALL messages
            # (including system messages with embedded dates/UUIDs/session IDs)
            stable, _ = _detector.extract_dynamic(cleaned_msg["content"])
            # Only replace if we actually found dynamic content
            if stable != cleaned_msg["content"]:
                cleaned_msg["content"] = stable
        cleaned.append(cleaned_msg)
    return cleaned


def _strip_attribution_block(text: str) -> str:
    """Remove attribution blocks that break cache prefix.

    Example attribution block:
        <antm:thinking_mode>...</antm:thinking_mode>
        <attribution>...</attribution>
    """
    import re
    # Strip common attribution patterns
    patterns = [
        r"<antm:[^>]*>.*?</antm:[^>]*>",
        r"<attribution>.*?</attribution>",
        r"<system_hint>.*?</system_hint>",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.DOTALL)
    return text.strip()


def _serialize_for_hash(messages: list[dict[str, Any]]) -> str:
    """Deterministic serialization for hashing."""
    return json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _extract_text(messages: list[dict[str, Any]]) -> str:
    """Extract all text content for rough token estimation."""
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
        # Include tool names in hash
        if "tool" in msg:
            parts.append(json.dumps(msg["tool"], sort_keys=True))
    return "".join(parts)
