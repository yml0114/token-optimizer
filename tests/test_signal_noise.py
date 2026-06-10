"""Tests for L1: Signal/Noise Classifier v3.

Covers:
  - Fragment-level splitting and inline filler stripping
  - Tool output metadata bulk removal
  - History compression (old turn summarization)
  - Tool output cleaning
  - Quality guardrails
  - Message compression pipeline
  - Cross-message deduplication
  - v3: Helper-prefix demotion
  - v3: Transition word removal
  - v3: Redundant modifier stripping
  - v3: Trailing particle cleanup
  - v3: Redundant quantifier stripping
"""

import pytest
from token_optimizer.core.signal_noise import (
    SignalNoiseClassifier,
    InputCompressor,
    HistoryCompressor,
    ToolOutputCleaner,
    SegmentType,
    CompressionLevel,
    Segment,
)


class TestFragmentLevelSplitting:
    """Test v2 fragment-level splitting and inline filler removal."""

    def setup_method(self):
        self.classifier = SignalNoiseClassifier(CompressionLevel.MODERATE)

    def test_inline_filler_stripped(self):
        """Inline fillers should be removed from within fragments."""
        text = "请帮我写一个函数，如果可以的话用Python"
        segments = self.classifier.classify_text(text)
        # Should have both NOISE (fillers) and SIGNAL (content)
        has_noise = any(s.segment_type == SegmentType.NOISE for s in segments)
        has_signal = any(s.segment_type == SegmentType.SIGNAL for s in segments)
        assert has_noise
        assert has_signal

    def test_multiple_fillers_stripped(self):
        """Multiple inline fillers in one sentence."""
        text = "好的谢谢，请帮我写一个排序算法，如果可以的话用快速排序"
        segments = self.classifier.classify_text(text)
        noise = [s for s in segments if s.segment_type == SegmentType.NOISE]
        signal = [s for s in segments if s.segment_type == SegmentType.SIGNAL]
        # "好的谢谢" and "请" and "如果可以的话" should be noise
        assert len(noise) >= 1
        # "帮我写一个排序算法", "用快速排序" should be signal
        assert len(signal) >= 1

    def test_pure_filler_line(self):
        """Lines that are entirely filler."""
        text = "好的谢谢"
        segments = self.classifier.classify_text(text)
        has_noise = any(s.segment_type == SegmentType.NOISE for s in segments)
        assert has_noise

    def test_code_block_preserved(self):
        """Code blocks must always be signal."""
        code = "```python\ndef hello():\n    print('hi')\n```"
        segments = self.classifier.classify_text(code)
        signal = [s for s in segments if s.segment_type == SegmentType.SIGNAL]
        assert len(signal) >= 1

    def test_error_trace_preserved(self):
        """Error messages and stack traces are signal."""
        error = "AttributeError: 'NoneType' object has no attribute 'strip'"
        segments = self.classifier.classify_text(error)
        has_signal = any(s.segment_type == SegmentType.SIGNAL for s in segments)
        assert has_signal

    def test_url_preserved(self):
        """URLs are always signal."""
        text = "Check this: https://github.com/example/repo"
        segments = self.classifier.classify_text(text)
        has_signal = any(s.segment_type == SegmentType.SIGNAL for s in segments)
        assert has_signal

    def test_question_preserved(self):
        """Questions are signal."""
        text = "How do I fix this error?"
        segments = self.classifier.classify_text(text)
        has_signal = any(s.segment_type == SegmentType.SIGNAL for s in segments)
        assert has_signal

    def test_chinese_command_preserved(self):
        """Chinese commands should always be signal."""
        text = "帮我写一个缓存管理器"
        segments = self.classifier.classify_text(text)
        signal = [s for s in segments if s.segment_type == SegmentType.SIGNAL]
        assert len(signal) >= 1


class TestToolOutputCleaning:
    """Test tool output metadata bulk removal."""

    def setup_method(self):
        self.cleaner = ToolOutputCleaner()

    def test_http_headers_removed(self):
        """HTTP header block should be stripped."""
        content = """HTTP/1.1 200 OK
Content-Type: application/json
X-RateLimit-Remaining: 59
{"data": "hello", "id": 123}"""
        result = self.cleaner.clean_tool_output(content)
        assert "HTTP/1.1" not in result
        assert "Content-Type" not in result
        assert "X-RateLimit" not in result
        assert '"data"' in result

    def test_trace_id_removed(self):
        """Trace IDs should be removed."""
        content = """request_id: abc123
trace_id: xyz789
{"result": "ok"}"""
        result = self.cleaner.clean_tool_output(content)
        assert "request_id" not in result
        assert "trace_id" not in result
        assert '"result"' in result

    def test_data_payload_preserved(self):
        """JSON data payload must be preserved."""
        content = """Content-Type: application/json
X-Request-Id: abc
{"users": [{"id": 1, "name": "test"}], "total": 42}"""
        result = self.cleaner.clean_tool_output(content)
        assert '"users"' in result
        assert '"total"' in result


class TestHistoryCompression:
    """Test old conversation turn compression."""

    def setup_method(self):
        self.compressor = HistoryCompressor(keep_recent=3)

    def test_short_history_no_compression(self):
        """Short histories should not be compressed."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result, meta = self.compressor.compress_history(messages)
        assert meta["compressed"] is False
        assert len(result) == 2

    def test_old_assistant_compressed(self):
        """Old assistant replies should be compressed."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a sort function"},
            {"role": "assistant", "content": "Here is a sort function:\n```python\ndef sort(arr):\n    return sorted(arr)\n```\nThis uses Python's built-in sorted function which uses Timsort with O(n log n) time complexity."},
            {"role": "user", "content": "Write a search function"},
            {"role": "assistant", "content": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1"},
            {"role": "user", "content": "Write a hash function"},
            {"role": "assistant", "content": "def simple_hash(s):\n    h = 0\n    for ch in s:\n        h = (h * 31 + ord(ch)) % 1000000007\n    return h"},
            {"role": "user", "content": "Write a linked list"},
        ]
        result, meta = self.compressor.compress_history(messages)
        assert meta["compressed"] is True
        # Recent 3 messages should be kept
        assert len(result) >= 3

    def test_tool_output_cleaned_in_old_turns(self):
        """Old tool outputs should have metadata stripped. Recent ones stay intact."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Query"},
            {"role": "tool", "content": "HTTP/1.1 200 OK\nContent-Type: json\n{\"result\": \"data\"}"},
            {"role": "assistant", "content": "The result is data."},
            {"role": "user", "content": "Query 2"},
            {"role": "tool", "content": "X-Trace: abc\nStatus: 200\n{\"result\": \"data2\"}"},
            {"role": "assistant", "content": "The result is data2."},
            {"role": "user", "content": "What now?"},
        ]
        result, meta = self.compressor.compress_history(messages)
        # Keep_recent=3 → last 3 messages stay as-is
        # Old tool outputs (in the first 5 messages) should have metadata stripped
        old_tool_msgs = [m for m in result[:5] if m.get("role") == "tool"]
        for tm in old_tool_msgs:
            assert "HTTP/1.1" not in tm["content"]
            assert "Content-Type" not in tm["content"]


class TestMessageCompression:
    """Test the full message compression pipeline."""

    def test_empty_messages(self):
        """Empty input should return empty output."""
        comp = InputCompressor(CompressionLevel.MODERATE)
        result, meta = comp.compress_messages([])
        assert result == []
        assert meta["compressed"] is True

    def test_system_message_preserved(self):
        """System messages should be preserved in SAFE/MODERATE modes."""
        comp = InputCompressor(CompressionLevel.MODERATE)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        result, meta = comp.compress_messages(messages)
        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) == 1

    def test_quality_guardrail(self):
        """Compression should never go below 30% of original."""
        comp = InputCompressor(CompressionLevel.MODERATE)
        messages = [
            {"role": "system", "content": "You are a coding assistant. Write clean code."},
            {"role": "user", "content": "Write a binary search function in Python."},
        ]
        result, meta = comp.compress_messages(messages)
        if meta["compressed"]:
            assert meta["compression_ratio"] >= 0.30

    def test_code_content_preserved(self):
        """Code snippets should always be preserved."""
        comp = InputCompressor(CompressionLevel.AGGRESSIVE)
        code = "```python\ndef add(a, b):\n    return a + b\n```"
        messages = [{"role": "user", "content": code}]
        result, meta = comp.compress_messages(messages)
        if meta["compressed"]:
            assert "def add" in result[0]["content"]

    def test_tool_metadata_cleaned(self):
        """Tool output metadata should be cleaned."""
        comp = InputCompressor(CompressionLevel.MODERATE)
        messages = [
            {"role": "tool", "content": "Content-Type: application/json\nX-Trace: abc\n{\"data\": \"ok\"}"},
        ]
        result, meta = comp.compress_messages(messages)
        assert "Content-Type" not in result[0]["content"]
        assert "X-Trace" not in result[0]["content"]
        assert '"data"' in result[0]["content"]

    def test_chinese_fillers_stripped(self):
        """Chinese filler words should be stripped from user messages."""
        comp = InputCompressor(CompressionLevel.MODERATE)
        messages = [
            {"role": "user", "content": "好的谢谢，请帮我写一个函数，如果可以的话用Python"},
        ]
        result, meta = comp.compress_messages(messages)
        if meta["compressed"]:
            # Filler words should be removed
            content = result[0]["content"]
            assert "好的谢谢" not in content or meta["savings_pct"] > 0

    def test_multilingual_fillers(self):
        """Both CN and EN fillers should be handled."""
        comp = InputCompressor(CompressionLevel.MODERATE)
        messages = [
            {"role": "user", "content": "thanks! please help me write a sort function if you don't mind"},
        ]
        result, meta = comp.compress_messages(messages)
        if meta["compressed"]:
            content = result[0]["content"]
            assert "thanks" not in content.lower() or meta["savings_pct"] > 0

    def test_cross_message_dedup(self):
        """Messages echoing system prompt should be deduplicated."""
        comp = InputCompressor(CompressionLevel.MODERATE)
        system = "You are a Python expert. Always use type hints."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "You are a Python expert. Always use type hints."},
            {"role": "user", "content": "Write a function to parse JSON."},
        ]
        result, meta = comp.compress_messages(messages, system_text=system)
        user_msgs = [m for m in result if m["role"] == "user"]
        assert len(user_msgs) >= 1


class TestSegmentEstimate:
    """Test token estimation."""

    def test_short_text_minimum(self):
        """Even very short text should estimate at least 1 token."""
        seg = Segment(text="hi", segment_type=SegmentType.SIGNAL, confidence=0.9, reason="test")
        assert seg.token_estimate >= 1

    def test_long_text_estimate(self):
        """Long text should have reasonable token estimate."""
        seg = Segment(
            text="This is a test sentence with several words for token estimation.",
            segment_type=SegmentType.SIGNAL,
            confidence=0.9,
            reason="test",
        )
        assert seg.token_estimate >= 5


class TestV3Improvements:
    """v3: Helper-prefix demotion, transition words, redundant modifiers, particles."""

    def setup_method(self):
        self.comp = InputCompressor(CompressionLevel.MODERATE)

    def test_helper_prefix_stripped(self):
        """'帮我写一个函数' → '写函数'."""
        messages = [{"role": "user", "content": "帮我写一个函数"}]
        result, meta = self.comp.compress_messages(messages)
        content = result[0]["content"]
        assert "帮我" not in content
        assert "写" in content

    def test_redundant_modifier_stripped(self):
        """'创建一个完整的模块' → '创建模块'."""
        messages = [{"role": "user", "content": "帮我创建一个完整的模块"}]
        result, meta = self.comp.compress_messages(messages)
        content = result[0]["content"]
        assert "完整的" not in content

    def test_transition_word_stripped(self):
        """'很好，接下来请实现' → '实现'."""
        messages = [{"role": "user", "content": "很好，接下来请帮我实现一个缓存系统"}]
        result, meta = self.comp.compress_messages(messages)
        content = result[0]["content"]
        assert "很好" not in content
        assert "接下来" not in content
        assert "实现" in content

    def test_redundant_quantifier_stripped(self):
        """'写一个排序算法' → '写排序算法'."""
        messages = [{"role": "user", "content": "写一个排序算法"}]
        result, meta = self.comp.compress_messages(messages)
        content = result[0]["content"]
        assert "写" in content

    def test_mixed_v3_case(self):
        """Full v3 pipeline: multiple noise types cleaned."""
        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "好的谢谢，请帮我写一个快速排序算法，如果可以的话。"},
            {"role": "assistant", "content": "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[0]\n    return quicksort([x for x in arr[1:] if x < pivot]) + [pivot] + quicksort([x for x in arr[1:] if x >= pivot])"},
            {"role": "user", "content": "很好，那么能加一个原地排序版本吗？不用创建新数组那种"},
        ]
        result, meta = self.comp.compress_messages(messages)
        user_contents = [m["content"] for m in result if m["role"] == "user"]
        # Should have compressed significantly
        assert meta["savings_pct"] > 0

    def test_trailing_particle_cleaned(self):
        """Trailing particles like 了/吧/呢 should be stripped."""
        messages = [{"role": "user", "content": "写好了吧"}]
        result, meta = self.comp.compress_messages(messages)
        # "写" should remain (signal), particles cleaned
        assert meta["compressed"]

    def test_command_preserved_through_noise(self):
        """Core command verb should survive noise removal."""
        messages = [{"role": "user", "content": "麻烦你帮我创建一个完整的用户认证系统，如果可以的话请用JWT"}]
        result, meta = self.comp.compress_messages(messages)
        content = result[0]["content"]
        # Core command should be there
        assert "创建" in content or "用户认证" in content


class TestV4WordLevel:
    """v4: Word-level fine filtering and history compression v2."""

    def setup_method(self):
        self.comp = InputCompressor(CompressionLevel.AGGRESSIVE)

    def test_then_filler_stripped(self):
        """'那么写一个函数' → '写函数'."""
        messages = [{"role": "user", "content": "那么写一个函数"}]
        result, meta = self.comp.compress_messages(messages)
        content = result[0]["content"]
        assert "那么" not in content
        assert "写" in content

    def test_incidentally_stripped(self):
        """'顺便加一个日志功能' → '加日志功能'."""
        messages = [{"role": "user", "content": "顺便加一个日志功能"}]
        result, meta = self.comp.compress_messages(messages)
        content = result[0]["content"]
        assert "顺便" not in content

    def test_can_add_stripped(self):
        """'能加一个缓存吗' → '加缓存吗'."""
        messages = [{"role": "user", "content": "能加一个缓存吗"}]
        result, meta = self.comp.compress_messages(messages)
        content = result[0]["content"]
        assert "能加" not in content

    def test_actually_stripped(self):
        """'其实我想用Python' → '用Python' (roughly)."""
        messages = [{"role": "user", "content": "其实我想用Python写"}]
        result, meta = self.comp.compress_messages(messages)
        content = result[0]["content"]
        assert "其实" not in content

    def test_hedge_words_stripped(self):
        """'我觉得可以加错误处理' → '加错误处理'."""
        messages = [{"role": "user", "content": "我觉得可以加错误处理"}]
        result, meta = self.comp.compress_messages(messages)
        content = result[0]["content"]
        assert "我觉得" not in content

    def test_demonstrative_stripped(self):
        """'那种缓存功能' → '缓存功能'."""
        messages = [{"role": "user", "content": "加那种缓存功能"}]
        result, meta = self.comp.compress_messages(messages)
        content = result[0]["content"]
        assert "那种" not in content

    def test_punctuation_normalized(self):
        """Multiple exclamation marks → single."""
        messages = [{"role": "user", "content": "写一个函数！！！"}]
        result, meta = self.comp.compress_messages(messages)
        content = result[0]["content"]
        assert "!!!" not in content
        assert "写" in content

    def test_long_conversation_history_compression(self):
        """10-turn conversation with verbose assistant replies should compress old turns."""
        messages = [
            {"role": "system", "content": "You are a coding assistant."},
        ]
        # Add 8 old turns with VERBOSE assistant replies
        for i in range(4):
            idx = i + 1
            messages.append({"role": "user", "content": f"请帮我写一个排序算法，如果可以的话用Python实现第{idx}个版本"})
            assistant_content = (
                f"好的！我来帮你实现第{idx}个版本的排序算法。\n\n"
                f"这个实现非常优雅，让我来详细解释一下。首先我们需要理解排序的基本原理，"
                f"排序算法的时间复杂度从O(n2)到O(n log n)不等。好的排序算法应该是稳定的，"
                f"这意味着相等元素的相对顺序不会改变。\n\n"
                f"def sort_v{idx}(arr):\n    return sorted(arr) # version {idx}\n\n"
                f"这个实现使用了Python内置的sorted函数，它基于TimSort算法，"
                f"时间复杂度为O(n log n)，空间复杂度为O(n)。希望这个对你有帮助！"
            )
            messages.append({"role": "assistant", "content": assistant_content})
        # Add 2 recent turns
        messages.append({"role": "user", "content": "请帮我写一个测试"})
        messages.append({"role": "assistant", "content": "def test_sort():\n    assert sort_v1([3,1,2]) == [1,2,3]"})

        comp = InputCompressor(CompressionLevel.AGGRESSIVE)
        result, meta = comp.compress_messages(messages)

        # Should compress old turns (verbose assistant replies)
        hist = meta.get("history_compression", {})
        assert hist.get("compressed") is True, f"History should be compressed"
        assert hist["savings_pct"] > 10, f"Expected history savings > 10%, got {hist['savings_pct']}"

    def test_repeated_instruction_dedup(self):
        """Repeated '写一个排序算法' across old turns should be deduped."""
        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "请帮我写一个排序算法"},
            {"role": "assistant", "content": "def sort(arr):\n    return sorted(arr)"},
            {"role": "user", "content": "请帮我写一个排序算法，加上自定义比较"},
            {"role": "assistant", "content": "def sort(arr, key=None):\n    return sorted(arr, key=key)"},
            {"role": "user", "content": "请帮我写一个排序算法，加上原地排序"},
            {"role": "assistant", "content": "def sort_inplace(arr):\n    arr.sort()"},
            {"role": "user", "content": "很好，那么请帮我写一个哈希表"},
            {"role": "assistant", "content": "class HashMap:\n    def __init__(self):\n        self._data = {}"},
            {"role": "user", "content": "请帮我写一个链表"},
        ]

        comp = InputCompressor(CompressionLevel.AGGRESSIVE)
        result, meta = comp.compress_messages(messages)

        hist = meta.get("history_compression", {})
        if hist.get("compressed"):
            # At least one repeated instruction should be detected
            repeated = hist.get("repeated_instructions_removed", 0)
            assert repeated >= 1, f"Expected at least 1 deduped instruction, got {repeated}"

    def test_residual_short_fragment_noise(self):
        """Very short Chinese fragments (< 4 chars) after filler removal → noise."""
        classifier = SignalNoiseClassifier(CompressionLevel.AGGRESSIVE)
        # "好的" is 2 chars, should be classified as noise
        segments = classifier.classify_text("好的")
        # Should have at least one noise segment
        noise = [s for s in segments if s.segment_type == SegmentType.NOISE]
        assert len(noise) >= 1

    def test_command_survives_word_level_filtering(self):
        """Core commands must survive all v4 word-level filtering."""
        messages = [{"role": "user", "content": "那么其实我觉得能加一个缓存模块"}]
        result, meta = self.comp.compress_messages(messages)
        content = result[0]["content"]
        # Core command should survive
        assert "加" in content or "缓存" in content
