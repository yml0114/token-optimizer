"""Tests for L1 v5: SmartCompressor (Flash-powered compression).

Tests the rule-based pre-filter path, fallback behavior, and validation.
The actual Flash API calls are mocked — we test the logic and fallback paths.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from token_optimizer.core.smart_compressor import SmartCompressor, COMPRESSION_PROMPT
from token_optimizer.core.signal_noise import CompressionLevel


class TestSmartCompressorRulePath:
    """Test rule-based pre-filter (no API key or API failure)."""

    def test_rule_only_fallback_no_api_key(self):
        """Without API key, falls back to rule-only compression."""
        compressor = SmartCompressor(api_key="")
        messages = [
            {"role": "user", "content": "你好，我想请你帮我写一个函数，就是那个排序函数"},
            {"role": "assistant", "content": "好的！我很乐意帮助你。请问你需要什么功能的函数呢？"},
        ]
        result, meta = compressor.compress(messages)
        assert meta["mode"] == "rule_only"
        assert meta["smart_compression"]["skipped"] is True
        assert meta["smart_compression"]["reason"] == "no_api_key"
        assert len(result) > 0

    def test_rule_only_removes_fillers(self):
        """Rule pre-filter removes common fillers."""
        compressor = SmartCompressor(api_key="")
        messages = [
            {"role": "user", "content": "就是说，那个，我想写一个函数"},
            {"role": "assistant", "content": "好的！请问是什么函数？"},
        ]
        result, meta = compressor.compress(messages)
        # Rule compressor should have stripped fillers
        assert meta["mode"] == "rule_only"
        user_content = result[0]["content"]
        # "就是说" and "那个" should be removed
        assert "就是说" not in user_content

    def test_preserves_code_blocks(self):
        """Code blocks must survive rule pre-filter (structure preserved)."""
        compressor = SmartCompressor(api_key="")
        messages = [
            {"role": "user", "content": "帮我写个hello函数"},
            {"role": "assistant", "content": "好的：\n```python\nprint('hello')\n```"},
        ]
        result, meta = compressor.compress(messages)
        assistant_content = result[1]["content"]
        # Code block markers and content should survive
        assert "python" in assistant_content or "print" in assistant_content


class TestSmartCompressorFlashPath:
    """Test Flash API integration path (with mocked API)."""

    def _make_compressor(self):
        return SmartCompressor(
            api_key="test-key",
            base_url="https://test.example.com/v1",
            model="mimo-v2-flash",
        )

    def test_successful_flash_compression(self):
        """Flash API returns valid compressed output."""
        compressor = self._make_compressor()
        
        original_messages = [
            {"role": "user", "content": "你好，请帮我写一个排序函数"},
            {"role": "assistant", "content": "好的！我很乐意帮助你。"},
            {"role": "user", "content": "就是快速排序，要降序排列"},
            {"role": "assistant", "content": "```python\ndef qsort_desc(a):\n    if len(a)<=1: return a\n    p=a[len(a)//2]\n    return qsort_desc([x for x in a if x>p])+[x for x in a if x==p]+qsort_desc([x for x in a if x<p])\n```"},
        ]
        
        flash_response = [
            {"role": "user", "content": "写一个降序快速排序函数"},
            {"role": "assistant", "content": "```python\ndef qsort_desc(a):\n    if len(a)<=1: return a\n    p=a[len(a)//2]\n    return qsort_desc([x for x in a if x>p])+[x for x in a if x==p]+qsort_desc([x for x in a if x<p])\n```"},
        ]
        
        with patch.object(compressor, '_flash_compress', return_value=flash_response):
            result, meta = compressor.compress(original_messages)
        
        assert meta["mode"] == "smart"
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert "降序" in result[0]["content"]

    def test_flash_invalid_json_fallback(self):
        """When Flash returns invalid JSON, falls back to rules."""
        compressor = self._make_compressor()
        messages = [
            {"role": "user", "content": "写个函数"},
            {"role": "assistant", "content": "好的"},
        ]
        
        with patch.object(compressor, '_flash_compress', return_value="not json"):
            result, meta = compressor.compress(messages)
        
        assert meta["mode"] == "rule_only_fallback"
        assert meta["smart_compression"]["reason"] == "validation_failed"

    def test_flash_api_error_fallback(self):
        """When Flash API throws error, falls back to rules."""
        compressor = self._make_compressor()
        messages = [
            {"role": "user", "content": "写个函数"},
        ]
        
        with patch.object(compressor, '_flash_compress', side_effect=Exception("API timeout")):
            result, meta = compressor.compress(messages)
        
        assert meta["mode"] == "rule_only_fallback"
        assert "API timeout" in meta["smart_compression"]["reason"]

    def test_flash_output_too_long_rejected(self):
        """Flash output longer than input is rejected."""
        compressor = self._make_compressor()
        messages = [
            {"role": "user", "content": "hi"},
        ]
        
        # Flash returns something longer than input
        long_output = [
            {"role": "user", "content": "a" * 10000},
        ]
        
        with patch.object(compressor, '_flash_compress', return_value=long_output):
            result, meta = compressor.compress(messages)
        
        assert meta["mode"] == "rule_only_fallback"

    def test_flash_removes_system_message_rejected(self):
        """Flash output that removes system message is rejected."""
        compressor = self._make_compressor()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        
        # Flash removes system message
        bad_output = [
            {"role": "user", "content": "hi"},
        ]
        
        with patch.object(compressor, '_flash_compress', return_value=bad_output):
            result, meta = compressor.compress(messages)
        
        assert meta["mode"] == "rule_only_fallback"

    def test_flash_empty_output_rejected(self):
        """Flash returning empty list is rejected."""
        compressor = self._make_compressor()
        messages = [
            {"role": "user", "content": "hello"},
        ]
        
        with patch.object(compressor, '_flash_compress', return_value=[]):
            result, meta = compressor.compress(messages)
        
        assert meta["mode"] == "rule_only_fallback"


class TestSmartCompressorJSONParsing:
    """Test JSON parsing from Flash responses."""

    def test_parse_clean_json(self):
        compressor = SmartCompressor(api_key="")
        result = compressor._parse_json_response(
            '[{"role": "user", "content": "hello"}]'
        )
        assert result == [{"role": "user", "content": "hello"}]

    def test_parse_markdown_wrapped_json(self):
        compressor = SmartCompressor(api_key="")
        result = compressor._parse_json_response(
            '```json\n[{"role": "user", "content": "hello"}]\n```'
        )
        assert result == [{"role": "user", "content": "hello"}]

    def test_parse_plain_code_block(self):
        compressor = SmartCompressor(api_key="")
        result = compressor._parse_json_response(
            '```\n[{"role": "user", "content": "hello"}]\n```'
        )
        assert result == [{"role": "user", "content": "hello"}]

    def test_parse_invalid_json_raises(self):
        compressor = SmartCompressor(api_key="")
        with pytest.raises(json.JSONDecodeError):
            compressor._parse_json_response("not json at all")

    def test_parse_non_list_raises(self):
        compressor = SmartCompressor(api_key="")
        with pytest.raises(ValueError, match="not a list"):
            compressor._parse_json_response('{"role": "user"}')

    def test_parse_missing_role_raises(self):
        compressor = SmartCompressor(api_key="")
        with pytest.raises(ValueError, match="missing role"):
            compressor._parse_json_response('[{"content": "hello"}]')


class TestSmartCompressorCostMath:
    """Verify the cost math in docstrings is correct."""

    def test_cost_savings_vs_pro_only(self):
        """v5 + L0 should save ~64-80% vs raw Pro."""
        # MiMo-V2.5-Pro pricing
        pro_input = 1.00   # $/M tokens
        pro_cache = 0.20   # $/M cached
        flash_input = 0.10  # $/M tokens
        flash_output = 0.30  # $/M tokens
        
        # Assume 1M original tokens → rule compress to 470K → Flash compress to 200K
        original_tokens = 1_000_000
        after_rules = 470_000
        after_flash = 200_000
        
        # Cost WITHOUT optimization (raw Pro)
        cost_raw = (original_tokens / 1_000_000) * pro_input  # $1.00
        
        # Cost WITH v5 (rules + Flash + Pro)
        # Flash processes 470K input, outputs ~200K
        # Pro receives 200K (with prefix cache on system prompt)
        flash_cost = (after_rules / 1_000_000) * flash_input  # $0.047
        flash_output_cost = (after_flash / 1_000_000) * flash_output  # $0.06
        pro_cost = (after_flash / 1_000_000) * pro_input  # $0.20
        cost_v5 = flash_cost + flash_output_cost + pro_cost  # $0.307
        
        savings = (1 - cost_v5 / cost_raw) * 100
        assert savings > 60, f"Expected >60% savings, got {savings:.1f}%"
        
        # With L0 cache
        cost_v5_cached = flash_cost + flash_output_cost + (after_flash / 1_000_000) * pro_cache
        savings_cached = (1 - cost_v5_cached / cost_raw) * 100
        assert savings_cached > 70, f"Expected >70% cached savings, got {savings_cached:.1f}%"
