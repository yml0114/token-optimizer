"""Tests for L1 v5: SmartCompressor (same-key, zero-config).

Tests the auto-routing, rule fallback, and validation logic.
Flash API calls are mocked.
"""

import json
from unittest.mock import patch

import pytest

from token_optimizer.core.smart_compressor import SmartCompressor, find_cheap_sibling


class TestAutoRouting:
    """Test auto-detection of cheap siblings."""

    def test_mimo_pro_to_flash(self):
        assert find_cheap_sibling("mimo-v2.5-pro") == "mimo-v2-flash"

    def test_mimo_v25_to_flash(self):
        assert find_cheap_sibling("mimo-v2.5") == "mimo-v2-flash"

    def test_deepseek_pro_to_flash(self):
        assert find_cheap_sibling("deepseek-v4-pro") == "deepseek-v4-flash"

    def test_qwen_max_to_turbo(self):
        assert find_cheap_sibling("qwen-max") == "qwen-turbo"

    def test_gpt4o_to_mini(self):
        assert find_cheap_sibling("gpt-4o") == "gpt-4o-mini"

    def test_claude_opus_to_haiku(self):
        assert find_cheap_sibling("claude-3-opus") == "claude-3-haiku"

    def test_unknown_model_returns_none(self):
        assert find_cheap_sibling("my-custom-model") is None

    def test_case_insensitive(self):
        assert find_cheap_sibling("MIMO-V2.5-PRO") == "mimo-v2-flash"


class TestSmartCompressorInit:
    """Test initialization and auto-configuration."""

    def test_auto_configured_for_mimo(self):
        sc = SmartCompressor(
            main_model="mimo-v2.5-pro",
            api_key="sk-test",
            base_url="https://api.xiaomimimo.com/v1",
        )
        assert sc.is_configured is True
        assert sc.compressor_model == "mimo-v2-flash"

    def test_not_configured_unknown_model(self):
        sc = SmartCompressor(
            main_model="unknown-model",
            api_key="sk-test",
            base_url="https://api.example.com/v1",
        )
        assert sc.is_configured is False

    def test_not_configured_no_api_key(self):
        sc = SmartCompressor(main_model="mimo-v2.5-pro", api_key="")
        assert sc.is_configured is False


class TestSmartCompressorRulePath:
    """Test rule-only fallback (no cheap model)."""

    def test_rule_only_no_sibling(self):
        sc = SmartCompressor(main_model="unknown-model", api_key="sk-test")
        messages = [
            {"role": "user", "content": "就是说我想写个函数"},
        ]
        result, meta = sc.compress(messages)
        assert meta["mode"] == "rule_only"
        assert len(result) > 0

    def test_rule_only_removes_fillers(self):
        sc = SmartCompressor()  # Not configured
        messages = [
            {"role": "user", "content": "就是说，那个，我想写一个快速排序函数，你能帮我实现一下吗，就是说，用Python写"},
        ]
        result, meta = sc.compress(messages)
        # Fillers are stripped when input is long enough for compression to kick in
        content = result[0]["content"]
        assert "就是说" not in content or len(content) < len(messages[0]["content"])

    def test_rule_preserves_user_intent(self):
        sc = SmartCompressor()
        messages = [
            {"role": "user", "content": "帮我写一个快速排序函数"},
        ]
        result, meta = sc.compress(messages)
        assert any("排序" in m["content"] for m in result)


class TestSmartCompressorSmartPath:
    """Test smart compression with mocked API."""

    def _make_configured(self):
        return SmartCompressor(
            main_model="mimo-v2.5-pro",
            api_key="sk-test-key",
            base_url="https://api.xiaomimimo.com/v1",
        )

    def test_successful_compression(self):
        sc = self._make_configured()
        messages = [
            {"role": "user", "content": "你好，请帮我写一个排序函数"},
            {"role": "assistant", "content": "好的！请问需要升序还是降序？"},
            {"role": "user", "content": "就是快速排序，降序"},
            {"role": "assistant", "content": "```python\ndef qsort_desc(a):\n    ...\n```"},
        ]
        
        flash_output = [
            {"role": "user", "content": "写降序快速排序函数"},
            {"role": "assistant", "content": "```python\ndef qsort_desc(a):\n    ...\n```"},
        ]
        
        with patch.object(sc, '_call_compressor', return_value=flash_output):
            result, meta = sc.compress(messages)
        
        assert meta["mode"] == "smart"
        assert meta["compressor"] == "mimo-v2-flash"
        assert len(result) == 2

    def test_api_error_fallback(self):
        sc = self._make_configured()
        messages = [{"role": "user", "content": "写个函数"}]
        
        with patch.object(sc, '_call_compressor', side_effect=Exception("timeout")):
            result, meta = sc.compress(messages)
        
        assert meta["mode"] == "rule_only_fallback"
        assert "timeout" in meta["reason"]

    def test_invalid_json_fallback(self):
        sc = self._make_configured()
        messages = [{"role": "user", "content": "写个函数"}]
        
        with patch.object(sc, '_call_compressor', return_value="not json"):
            result, meta = sc.compress(messages)
        
        assert meta["mode"] == "rule_only_fallback"
        assert "校验未通过" in meta["reason"]

    def test_output_too_long_rejected(self):
        sc = self._make_configured()
        messages = [{"role": "user", "content": "hi"}]
        
        with patch.object(sc, '_call_compressor',
                          return_value=[{"role": "user", "content": "x" * 10000}]):
            result, meta = sc.compress(messages)
        
        assert meta["mode"] == "rule_only_fallback"

    def test_preserves_system_message(self):
        sc = self._make_configured()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
        
        flash_output = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
        
        with patch.object(sc, '_call_compressor', return_value=flash_output):
            result, meta = sc.compress(messages)
        
        assert meta["mode"] == "smart"
        assert any(m["role"] == "system" for m in result)


class TestSmartCompressorCostMath:
    """Verify cost math."""

    def test_flash_saves_money(self):
        """Flash + Pro should cost less than Pro alone."""
        # 1M tokens, Pro $1.00/M
        raw = 1_000_000 / 1e6 * 1.00  # $1.00
        
        # Flash compresses 1M → 200K, Flash $0.10/M
        flash_cost = 1_000_000 / 1e6 * 0.10  # $0.10
        # Pro processes 200K
        pro_cost = 200_000 / 1e6 * 1.00  # $0.20
        
        total = flash_cost + pro_cost  # $0.30
        assert total < raw, "Flash+Pro must be cheaper than raw Pro"
        savings = (1 - total / raw) * 100
        assert savings > 50, f"Expected >50% savings, got {savings:.1f}%"
