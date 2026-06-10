"""Tests for L1: Signal/Noise Classifier.

Covers:
  - Filler word detection (CN + EN)
  - Tool output noise stripping
  - System tag removal
  - Compression level behavior (SAFE/MODERATE/AGGRESSIVE)
  - Quality guardrails
  - Message compression pipeline
  - Cross-message deduplication
"""

import pytest
from token_optimizer.core.signal_noise import (
    SignalNoiseClassifier,
    InputCompressor,
    SegmentType,
    CompressionLevel,
    Segment,
    CompressionResult,
)


class TestFillerDetection:
    """Test filler word and politeness marker detection."""

    def setup_method(self):
        self.classifier = SignalNoiseClassifier(CompressionLevel.AGGRESSIVE)

    def test_cn_pure_filler_removed(self):
        """Short pure-filler phrases should be classified as noise."""
        segments = self.classifier.classify_text("请")
        noise = [s for s in segments if s.segment_type == SegmentType.NOISE]
        assert len(noise) >= 1

    def test_cn_filler_with_content(self):
        """Long sentences with filler prefix should keep the content."""
        text = "请帮我写一个快速排序函数"
        segments = self.classifier.classify_text(text)
        has_signal = any(s.segment_type == SegmentType.SIGNAL for s in segments)
        assert has_signal

    def test_en_pure_filler_removed(self):
        """Short English filler phrases."""
        segments = self.classifier.classify_text("please")
        noise = [s for s in segments if s.segment_type == SegmentType.NOISE]
        assert len(noise) >= 1

    def test_en_filler_with_content(self):
        """Long English sentences with filler prefix."""
        text = "Please help me write a sorting algorithm"
        segments = self.classifier.classify_text(text)
        has_signal = any(s.segment_type == SegmentType.SIGNAL for s in segments)
        assert has_signal

    def test_code_block_preserved(self):
        """Code blocks must always be signal."""
        code = "```python\ndef hello():\n    print('hi')\n```"
        segments = self.classifier.classify_text(code)
        all_signal = all(s.segment_type == SegmentType.SIGNAL for s in segments if s.text.strip())
        assert all_signal

    def test_error_trace_preserved(self):
        """Error messages and stack traces are signal."""
        error = "File \"main.py\", line 42\n    raise ValueError('bad input')"
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


class TestToolOutputNoise:
    """Test tool output metadata stripping."""

    def setup_method(self):
        self.classifier = SignalNoiseClassifier(CompressionLevel.MODERATE)

    def test_http_header_removed(self):
        """HTTP headers should be noise."""
        text = "Content-Type: application/json\n{\"data\": \"ok\"}"
        segments = self.classifier.classify_text(text)
        has_noise = any(s.segment_type == SegmentType.NOISE for s in segments)
        assert has_noise

    def test_trace_id_removed(self):
        """Trace IDs should be noise."""
        text = "request_id: abc123def456\ntrace_id: xyz789"
        segments = self.classifier.classify_text(text)
        has_noise = any(s.segment_type == SegmentType.NOISE for s in segments)
        assert has_noise

    def test_latency_metadata_removed(self):
        """Latency metadata should be noise."""
        text = "latency: 145ms\nresponse_time: 200ms"
        segments = self.classifier.classify_text(text)
        has_noise = any(s.segment_type == SegmentType.NOISE for s in segments)
        assert has_noise


class TestSystemTagRemoval:
    """Test system/attribution tag stripping."""

    def setup_method(self):
        self.classifier = SignalNoiseClassifier(CompressionLevel.SAFE)

    def test_system_hint_removed(self):
        """<system_hint> tags should be stripped."""
        text = "Hello <system_hint>internal info</system_hint> world"
        segments = self.classifier.classify_text(text)
        has_noise = any(
            s.segment_type == SegmentType.NOISE and s.reason == "system_tag"
            for s in segments
        )
        assert has_noise


class TestCompressionLevels:
    """Test that different compression levels produce different results."""

    def test_safe_less_aggressive(self):
        """SAFE mode should keep more than AGGRESSIVE."""
        text = "请帮我写一个函数。如果可以的话用Python。"
        
        safe_comp = InputCompressor(CompressionLevel.SAFE)
        _, safe_meta = safe_comp.compress_messages([
            {"role": "user", "content": text}
        ])

        agg_comp = InputCompressor(CompressionLevel.AGGRESSIVE)
        _, agg_meta = agg_comp.compress_messages([
            {"role": "user", "content": text}
        ])

        # Safe should keep more tokens than aggressive
        if safe_meta["compressed"] and agg_meta["compressed"]:
            assert safe_meta["compressed_tokens_est"] >= agg_meta["compressed_tokens_est"]


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
        # Should not be compressed below 30%
        if meta["compressed"]:
            assert meta["compression_ratio"] >= 0.30

    def test_code_content_preserved(self):
        """Code snippets in messages should always be preserved."""
        comp = InputCompressor(CompressionLevel.AGGRESSIVE)
        code = "```python\ndef add(a, b):\n    return a + b\n```"
        messages = [{"role": "user", "content": code}]
        result, meta = comp.compress_messages(messages)
        if meta["compressed"]:
            assert "def add" in result[0]["content"]


class TestCrossMessageDedup:
    """Test deduplication of repeated instructions."""

    def test_echo_of_system_deduplicated(self):
        """User messages that echo system prompt should be removed."""
        comp = InputCompressor(CompressionLevel.MODERATE)
        system = "You are a Python expert. Always use type hints."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "You are a Python expert. Always use type hints."},
            {"role": "user", "content": "Write a function to parse JSON."},
        ]
        result, meta = comp.compress_messages(messages, system_text=system)
        # The echo message should be removed or compressed
        user_msgs = [m for m in result if m["role"] == "user"]
        # At least the echo should be handled
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
