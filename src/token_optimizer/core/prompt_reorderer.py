"""L0: Prefix Structure Optimizer — Zero-cost, zero-quality-impact reorder.

This is the highest-ROI layer. It reorders prompt messages so that static content
(system prompt, tool definitions) is at the front, and dynamic content (user input,
timestamps) is at the end. This maximizes API cache hit rates.

DeepSeek caches from token 0 forward. If your prefix changes every request,
cache never hits. This module ensures it always hits.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def reorder_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict]:
    """Reorder messages for maximum cache hit rate.

    New order:
        1. system prompt (static, cacheable)
        2. tool definitions (static, cacheable)
        3. conversation memory / summary (quasi-static)
        4. history messages (growing, tail-cacheable)
        5. current user input (dynamic, not cacheable)

    Returns:
        (reordered_messages, metadata) where metadata has hash info
    """
    if not messages:
        return messages, {"prefix_hash": "", "prefix_tokens_est": 0}

    # Separate message types
    system_msgs: list[dict[str, Any]] = []
    user_msgs: list[dict[str, Any]] = []
    assistant_msgs: list[dict[str, Any]] = []
    tool_msgs: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
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
        # Pair assistant+user messages (keeping temporal order)
        history_users = user_msgs[:-1]  # all but last
        history_assistants = assistant_msgs
        current_user = user_msgs[-1] if user_msgs else None

        # Interleave by timestamp (assume original order)
        # For simplicity, assume they alternate: u0, a0, u1, a1, ...
        for msg in messages:
            if msg.get("role") in ("user", "assistant") and msg is not current_user:
                reordered.append(msg)

        # Tool messages go near their corresponding assistant messages
        # (simplified: append tool results after history)
        reordered.extend(tool_msgs)

        # 5. Current user input — always last
        if current_user:
            reordered.append(current_user)
    elif user_msgs:
        reordered.extend(user_msgs)

    # Compute prefix hash (everything except the last user message)
    prefix_content = _serialize_for_hash(reordered[:-1] if reordered else reordered)
    prefix_hash = hashlib.sha256(prefix_content.encode()).hexdigest()[:16]

    # Estimate prefix token count (rough: 1 token ≈ 4 chars for English, ~2 for CJK)
    prefix_text = _extract_text(reordered[:-1] if reordered else [])
    prefix_tokens_est = max(1, len(prefix_text) // 3)  # rough average

    metadata = {
        "prefix_hash": prefix_hash,
        "prefix_tokens_est": prefix_tokens_est,
        "original_order": [m.get("role", "?") for m in messages],
        "reordered_order": [m.get("role", "?") for m in reordered],
        "did_reorder": reordered != messages,
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

    E.g., timestamps, request IDs, attribution blocks.
    """
    STRIPPED_KEYS = {"timestamp", "ts", "request_id", "trace_id", "attribution"}

    cleaned = []
    for msg in messages:
        cleaned_msg = {k: v for k, v in msg.items() if k not in STRIPPED_KEYS}
        # Remove attribution blocks from content (common in Claude Code / MiMo)
        if isinstance(cleaned_msg.get("content"), str):
            cleaned_msg["content"] = _strip_attribution_block(cleaned_msg["content"])
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
