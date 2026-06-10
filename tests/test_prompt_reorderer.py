"""Tests for L0: Prompt Reorderer."""

import pytest
from token_optimizer.core.prompt_reorderer import reorder_messages, compute_prefix_hash, strip_dynamic_fields


class TestReorderMessages:
    def test_basic_reorder(self):
        """System prompt should move to front, user message to end."""
        messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        reordered, meta = reorder_messages(messages)
        assert reordered[0]["role"] == "system"
        assert reordered[-1]["role"] == "user"
        assert meta["did_reorder"] is True

    def test_already_optimal_order(self):
        """If already in optimal order, should not change."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        reordered, meta = reorder_messages(messages)
        assert len(reordered) == 2
        assert meta["did_reorder"] is False

    def test_tools_sorted(self):
        """Tools should be sorted alphabetically."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        tools = [
            {"function": {"name": "search_web"}},
            {"function": {"name": "calculate"}},
        ]
        reordered, meta = reorder_messages(messages, tools)
        tool_msgs = [m for m in reordered if m.get("role") == "tool_def"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool"]["function"]["name"] == "calculate"
        assert tool_msgs[1]["tool"]["function"]["name"] == "search_web"

    def test_preserves_multiple_system(self):
        """Multiple system messages should all stay at front."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
        ]
        reordered, meta = reorder_messages(messages)
        assert reordered[0]["role"] == "system"
        assert reordered[1]["role"] == "system"
        assert reordered[-1]["role"] == "user"

    def test_compute_prefix_hash_stability(self):
        """Same prefix should produce same hash."""
        msgs1 = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        msgs2 = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        h1 = compute_prefix_hash(msgs1)
        h2 = compute_prefix_hash(msgs2)
        assert h1 == h2

    def test_prefix_hash_changes_with_content(self):
        """Different prefix should produce different hash."""
        msgs1 = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        msgs2 = [
            {"role": "system", "content": "You are evil."},
            {"role": "user", "content": "Hello!"},
        ]
        h1 = compute_prefix_hash(msgs1)
        h2 = compute_prefix_hash(msgs2)
        assert h1 != h2

    def test_ignore_last_user_in_hash(self):
        """Only the prefix (not last user msg) should be hashed."""
        msgs1 = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Question 1?"},
        ]
        msgs2 = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Question 2?"},
        ]
        # Same prefix (system prompt), different last user msg → same hash
        h1 = compute_prefix_hash(msgs1)
        h2 = compute_prefix_hash(msgs2)
        assert h1 == h2


class TestStripDynamicFields:
    def test_strip_timestamp(self):
        """Timestamps should be removed."""
        msg = {"role": "user", "content": "Hello", "timestamp": "2026-01-01T00:00:00Z"}
        cleaned = strip_dynamic_fields([msg])
        assert "timestamp" not in cleaned[0]

    def test_strip_attribution_block(self):
        """Attribution blocks in content should be stripped."""
        msg = {
            "role": "assistant",
            "content": "Hello<attribution>This is attribution</attribution>World",
        }
        cleaned = strip_dynamic_fields([msg])
        assert "<attribution>" not in cleaned[0]["content"]
        assert "HelloWorld" in cleaned[0]["content"]

    def test_preserve_critical_content(self):
        """Critical message fields should be preserved."""
        msg = {
            "role": "user",
            "content": "Important question",
            "name": "test_user",
        }
        cleaned = strip_dynamic_fields([msg])
        assert cleaned[0]["content"] == "Important question"
        assert cleaned[0]["name"] == "test_user"
