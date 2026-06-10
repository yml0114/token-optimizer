"""Tests for L1 v5: SmartCompressor (same-key, zero-config).

Tests the auto-routing, rule fallback, and validation logic.
Flash API calls are mocked.
"""

import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from token_optimizer.core.smart_compressor import (
    CheapModelOption,
    ModelRoute,
    SmartCompressor,
    estimate_tokens_from_text,
    find_cheap_sibling,
)


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
            min_rule_tokens_for_smart=1,
        )

    def test_successful_compression(self):
        sc = self._make_configured()
        messages = [
            {"role": "user", "content": "你好，请帮我写一个排序函数，要求快速排序，降序，Python实现" * 20},
            {"role": "assistant", "content": "好的，我会给你 Python 快速排序降序实现。" * 10},
            {"role": "user", "content": "请只保留核心代码，不需要解释" * 10},
        ]
        
        flash_output = [
            {"role": "user", "content": "Python降序快速排序核心代码"},
        ]
        
        with patch.object(sc, '_call_compressor', return_value=flash_output):
            result, meta = sc.compress(messages)
        
        assert meta["mode"] == "smart"
        assert meta["compressor"] == "mimo-v2-flash"
        assert len(result) == 1
        assert meta["profit_guard"]["actual"]["profitable"] is True

    def test_api_error_fallback(self):
        sc = self._make_configured()
        messages = [{"role": "user", "content": "请帮我写一个函数，并解释边界条件和测试用例" * 40}]
        
        with patch.object(sc, '_call_compressor', side_effect=Exception("timeout")):
            result, meta = sc.compress(messages)
        
        assert meta["mode"] == "rule_only_fallback"
        assert "timeout" in meta["reason"]

    def test_invalid_json_fallback(self):
        sc = self._make_configured()
        messages = [{"role": "user", "content": "请帮我写一个函数，并解释边界条件和测试用例" * 40}]
        
        with patch.object(sc, '_call_compressor', return_value="not json"):
            result, meta = sc.compress(messages)
        
        assert meta["mode"] == "rule_only_fallback"
        assert "校验未通过" in meta["reason"]

    def test_output_too_long_rejected(self):
        sc = self._make_configured()
        messages = [{"role": "user", "content": "请帮我写一个函数，并解释边界条件和测试用例" * 40}]
        
        with patch.object(sc, '_call_compressor',
                          return_value=[{"role": "user", "content": "x" * 10000}]):
            result, meta = sc.compress(messages)
        
        assert meta["mode"] == "rule_only_fallback"

    def test_preserves_system_message(self):
        sc = self._make_configured()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "请帮我总结这段长内容" * 80},
        ]
        
        flash_output = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "总结长内容"},
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

    def test_short_input_skips_smart_compression(self):
        """Tiny inputs should not pay the extra cheap-model call."""
        sc = SmartCompressor(
            main_model="mimo-v2.5-pro",
            api_key="sk-test-key",
            base_url="https://api.xiaomimimo.com/v1",
        )
        with patch.object(sc, '_call_compressor', side_effect=AssertionError("should not call cheap model")):
            result, meta = sc.compress([{"role": "user", "content": "hi"}])
        assert meta["mode"] == "rule_only_profit_guard"
        assert "输入过短" in meta["reason"]

    def test_unprofitable_actual_result_falls_back(self):
        """If cheap model fails to compress enough, keep rule-only result."""
        sc = SmartCompressor(
            main_model="mimo-v2.5-pro",
            api_key="sk-test-key",
            base_url="https://api.xiaomimimo.com/v1",
            min_rule_tokens_for_smart=1,
        )
        messages = [{"role": "user", "content": "请帮我总结这个项目" * 100}]
        too_long = [{"role": "user", "content": "请帮我总结这个项目" * 90}]

        with patch.object(sc, '_call_compressor', return_value=too_long):
            result, meta = sc.compress(messages)

        assert meta["mode"] == "rule_only_profit_guard"
        assert "收益不足" in meta["reason"]
        assert meta["profit_guard"]["actual"]["profitable"] is False

    def test_context_guard_skips_cheap_model_when_input_exceeds_window(self):
        """Cheap model must not be called when rule-compressed input exceeds its context."""
        sc = SmartCompressor(
            main_model="mimo-v2.5-pro",
            api_key="sk-test-key",
            base_url="https://api.xiaomimimo.com/v1",
            min_rule_tokens_for_smart=1,
        )
        # Use a per-instance route copy, not a global ROUTES mutation.
        tiny_option = replace(sc.route.cheap_options[0], max_context=2)
        sc.route = replace(sc.route, cheap_options=(tiny_option,))
        sc.active_option = tiny_option
        sc.compressor_model = tiny_option.model
        with patch.object(sc, '_call_compressor', side_effect=AssertionError("should not call cheap model")):
            result, meta = sc.compress([{"role": "user", "content": "这是一个明显超过两个token的长输入"}])
        assert meta["mode"] == "rule_only_context_guard"
        assert "上下文窗口" in meta["reason"]


class TestSmartCompressorTokenizerAndRouting:
    """Verify stronger token estimation and multi-candidate routing."""

    def test_multilingual_token_estimator_distinguishes_cjk_and_ascii(self):
        chinese = estimate_tokens_from_text("这是一个用于压缩测试的中文长句子")
        english = estimate_tokens_from_text("this is an english compression test sentence")
        assert chinese > 5
        assert english > 5
        assert chinese != len("这是一个用于压缩测试的中文长句子") // 3

    def test_multi_candidate_router_picks_best_profitable_option(self):
        sc = SmartCompressor(
            main_model="custom-pro",
            api_key="sk-test-key",
            base_url="https://api.example.com/v1",
            min_rule_tokens_for_smart=1,
        )
        expensive = CheapModelOption("custom-expensive-compressor", 0.90, 2.00, max_context=1_000_000)
        cheap = CheapModelOption("custom-cheap-compressor", 0.01, 0.02, max_context=1_000_000)
        sc.route = ModelRoute(
            pattern="custom-pro",
            main_input_price=1.00,
            main_output_price=3.00,
            cheap_options=(expensive, cheap),
        )
        sc.active_option = expensive
        sc.compressor_model = expensive.model
        sc.is_configured = True

        messages = [{"role": "user", "content": "请压缩这段很长的项目上下文" * 100}]
        flash_output = [{"role": "user", "content": "压缩项目上下文"}]

        with patch.object(sc, '_call_compressor', return_value=flash_output):
            result, meta = sc.compress(messages)

        assert meta["mode"] == "smart"
        assert meta["compressor"] == "custom-cheap-compressor"
        assert meta["route"]["selected_candidate"] == "custom-cheap-compressor"
        assert meta["profit_guard"]["projected"]["candidate"] == "custom-cheap-compressor"
