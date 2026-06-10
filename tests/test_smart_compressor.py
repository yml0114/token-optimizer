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
    assess_compression_policy,
    estimate_tokens_from_text,
    extract_protected_spans,
    find_cheap_sibling,
    format_protected_spans,
    score_semantic_fidelity,
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


class TestSmartCompressorSelfLearningRepair:
    """Verify learning feedback and self-repair cost guards."""

    def _make_configured(self):
        return SmartCompressor(
            main_model="mimo-v2.5-pro",
            api_key="sk-test-key",
            base_url="https://api.xiaomimimo.com/v1",
            min_rule_tokens_for_smart=1,
            max_consecutive_failures=2,
            circuit_breaker_cooldown=10,
        )

    def test_learning_updates_expected_ratio_after_success(self):
        sc = self._make_configured()
        messages = [{"role": "user", "content": "请总结这段长项目上下文" * 120}]
        short_output = [{"role": "user", "content": "项目上下文摘要"}]

        before = sc._project_profit(300, sc.active_option)["expected_smart_ratio"]
        with patch.object(sc, '_call_compressor', return_value=short_output):
            result, meta = sc.compress(messages)
        after = sc._project_profit(300, sc.active_option)["expected_smart_ratio"]

        assert meta["mode"] == "smart"
        assert meta["profit_guard"]["learning"]["mimo-v2-flash"]["successes"] == 1
        assert after < before

    def test_circuit_breaker_stops_repeated_broken_compressor_calls(self):
        sc = self._make_configured()
        messages = [{"role": "user", "content": "请总结这段长项目上下文" * 120}]
        too_long = [{"role": "user", "content": "请总结这段长项目上下文" * 100}]

        with patch.object(sc, '_call_compressor', return_value=too_long) as mocked_call:
            _, meta1 = sc.compress(messages)
            _, meta2 = sc.compress(messages)
        assert mocked_call.call_count == 2
        assert meta1["mode"] == "rule_only_profit_guard"
        assert meta2["profit_guard"]["learning"]["mimo-v2-flash"]["consecutive_failures"] == 2

        with patch.object(sc, '_call_compressor', side_effect=AssertionError("circuit should prevent API cost")):
            _, meta3 = sc.compress(messages)
        assert meta3["mode"] == "rule_only_self_repair"
        assert "熔断" in meta3["reason"]
        assert meta3["profit_guard"]["candidate_diagnostics"][0]["blocked_by_circuit"] is True

    def test_rule_compressor_exception_passthrough_without_api_call(self):
        sc = self._make_configured()
        messages = [{"role": "user", "content": "这个请求不能因为压缩崩溃而失败"}]
        with patch.object(sc.rule_compressor, 'compress_messages', side_effect=RuntimeError("boom")):
            with patch.object(sc, '_call_compressor', side_effect=AssertionError("should not call cheap model")):
                result, meta = sc.compress(messages)
        assert result == messages
        assert meta["mode"] == "safe_passthrough_repair"
        assert "未调用廉价模型" in meta["reason"]



class TestSmartCompressorDualThresholdPolicy:
    """Verify safe/extreme/protected dual-threshold policy."""

    def test_policy_extreme_for_low_risk_noise(self):
        policy = assess_compression_policy([
            {"role": "assistant", "content": "当然可以，谢谢你的补充。就是说这个历史背景大概可以总结一下。" * 8},
            {"role": "tool", "content": '{"status":"ok","metadata":{"trace_id":"abc","file_size":123}}'},
        ])
        assert policy.mode == "extreme"
        assert policy.target_ratio == 0.22

    def test_policy_protected_for_code_and_errors(self):
        policy = assess_compression_policy([
            {"role": "user", "content": "Traceback: ValueError at /app/main.py line 42, def run(): return price * 0.2"},
        ])
        assert policy.mode == "protected"
        assert policy.target_ratio == 0.45

    def test_policy_metadata_exposed_in_compress_result(self):
        sc = SmartCompressor(
            main_model="mimo-v2.5-pro",
            api_key="sk-test-key",
            base_url="https://api.xiaomimimo.com/v1",
            min_rule_tokens_for_smart=1,
        )
        messages = [{"role": "user", "content": "谢谢，就是说这个历史背景大概总结一下" * 20}]
        flash_output = [{"role": "user", "content": "总结历史背景"}]
        with patch.object(sc, '_call_compressor', return_value=flash_output):
            result, meta = sc.compress(messages)
        assert meta["mode"] == "smart"
        assert meta["compression_policy"]["mode"] in {"safe", "extreme"}
        assert "target_ratio" in meta["compression_policy"]



class TestSmartCompressorFidelityGuard:
    """Verify compression cannot drop critical task signals."""

    def _make_configured(self):
        return SmartCompressor(
            main_model="mimo-v2.5-pro",
            api_key="sk-test-key",
            base_url="https://api.xiaomimimo.com/v1",
            min_rule_tokens_for_smart=1,
        )

    def test_fidelity_guard_rejects_missing_numbers_paths_and_url(self):
        sc = self._make_configured()
        messages = [{
            "role": "user",
            "content": (
                "请修复 /app/data/project/main.py 的 parse_price()，错误码 500，"
                "金额必须保持 ¥19.9，接口 https://api.example.com/v1/prices，不要改参数。"
            ) * 20,
        }]
        lossy_output = [{"role": "user", "content": "修复价格解析函数，保持接口参数。"}]
        with patch.object(sc, '_call_compressor', return_value=lossy_output):
            result, meta = sc.compress(messages)
        assert meta["mode"] == "rule_only_fidelity_guard"
        assert meta["fidelity_guard"]["passed"] is False
        assert "paths" in meta["fidelity_guard"]["missing"]
        assert "urls" in meta["fidelity_guard"]["missing"]

    def test_fidelity_guard_allows_preserved_critical_signals(self):
        sc = self._make_configured()
        messages = [{
            "role": "user",
            "content": (
                "请修复 /app/data/project/main.py 的 parse_price()，错误码 500，"
                "金额必须保持 ¥19.9，接口 https://api.example.com/v1/prices，不要改参数。"
            ) * 20,
        }]
        safe_output = [{
            "role": "user",
            "content": "修复 /app/data/project/main.py 的 parse_price()；保留错误码 500、金额 ¥19.9、接口 https://api.example.com/v1/prices，参数不改。",
        }]
        with patch.object(sc, '_call_compressor', return_value=safe_output):
            result, meta = sc.compress(messages)
        assert meta["mode"] == "smart"
        assert meta["fidelity_guard"]["passed"] is True
        assert meta["fidelity_guard"]["score"] >= meta["fidelity_guard"]["threshold"]

    def test_score_semantic_fidelity_direct_api(self):
        original = [{"role": "user", "content": "保留 order_id=12345 和 /tmp/a.json"}]
        compressed = [{"role": "user", "content": "保留订单和文件"}]
        report = score_semantic_fidelity(original, compressed)
        assert report.passed is False
        assert report.score < report.threshold


class TestProtectedSpans:
    """Verify deterministic protected-span extraction for smart compression."""

    def test_extract_protected_spans_covers_hard_signals(self):
        messages = [{
            "role": "user",
            "content": (
                "修复 /app/data/project/main.py 的 parse_price()，错误码 500，"
                "接口 https://api.example.com/v1/prices，邮箱 support@unfaze.app。"
            ),
        }]
        spans = extract_protected_spans(messages)
        values = {span.value for span in spans}
        kinds = {span.kind for span in spans}
        assert "/app/data/project/main.py" in values
        assert "https://api.example.com/v1/prices" in values
        assert "support@unfaze.app" in values
        assert "parse_price()" in values
        assert "500" in values
        assert {"paths", "urls", "emails", "code_symbols", "numbers"}.issubset(kinds)

    def test_format_protected_spans_for_prompt(self):
        spans = extract_protected_spans([{
            "role": "user",
            "content": "保留 /tmp/a.json 和 request_id=req_123，金额 ¥19.9。",
        }])
        text = format_protected_spans(spans)
        assert "PROTECTED_SPANS" in text
        assert "/tmp/a.json" in text
        assert "¥19.9" in text

    def test_metadata_exposes_protected_spans(self):
        sc = SmartCompressor(
            main_model="mimo-v2.5-pro",
            api_key="sk-test-key",
            base_url="https://api.xiaomimimo.com/v1",
            min_rule_tokens_for_smart=1,
        )
        messages = [{
            "role": "user",
            "content": "修复 /app/data/project/main.py 的 parse_price()，错误码 500。" * 20,
        }]
        safe_output = [{"role": "user", "content": "修复 /app/data/project/main.py 的 parse_price()，保留错误码 500。"}]
        with patch.object(sc, '_call_compressor', return_value=safe_output):
            _result, meta = sc.compress(messages)
        assert meta["protected_spans"]["count"] >= 3
        protected_values = {item["value"] for item in meta["protected_spans"]["items"]}
        assert "/app/data/project/main.py" in protected_values
        assert "parse_price()" in protected_values
        assert "500" in protected_values

    def test_call_payload_includes_protected_spans(self):
        sc = SmartCompressor(
            main_model="mimo-v2.5-pro",
            api_key="sk-test-key",
            base_url="https://api.xiaomimimo.com/v1",
            min_rule_tokens_for_smart=1,
        )
        messages = [{
            "role": "user",
            "content": "修复 /app/data/project/main.py 的 parse_price()，错误码 500。" * 20,
        }]
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [{
                        "message": {
                            "content": json.dumps([{"role": "user", "content": "修复 /app/data/project/main.py 的 parse_price()，保留错误码 500。"}], ensure_ascii=False),
                        },
                    }],
                }

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, json=None, headers=None):
                captured["payload"] = json
                return FakeResponse()

        with patch("token_optimizer.core.smart_compressor.httpx.Client", FakeClient):
            _result, meta = sc.compress(messages)

        user_prompt = captured["payload"]["messages"][1]["content"]
        assert "PROTECTED_SPANS" in user_prompt
        assert "/app/data/project/main.py" in user_prompt
        assert "parse_price()" in user_prompt
        assert "500" in user_prompt
        assert meta["mode"] == "smart"



class TestSmartCompressorShadowTelemetry:
    """Verify shadow mode telemetry does not affect real request path."""

    def _make_configured(self, **kwargs):
        return SmartCompressor(
            main_model="mimo-v2.5-pro",
            api_key="sk-test-key",
            base_url="https://api.xiaomimimo.com/v1",
            min_rule_tokens_for_smart=1,
            **kwargs,
        )

    def test_shadow_does_not_call_cheap_model(self):
        sc = self._make_configured()
        messages = [{"role": "user", "content": "请总结这段很长的项目上下文" * 120}]
        with patch.object(sc, '_call_compressor', side_effect=AssertionError("shadow must not call cheap model")):
            telemetry = sc.shadow_evaluate(messages)
        assert telemetry["mode"] == "shadow"
        assert telemetry["would_call_smart"] is True
        assert telemetry["selected_candidate"] == "mimo-v2-flash"
        assert telemetry["estimated_smart_tokens"] is not None
        assert telemetry["estimated_savings_pct"] > 0
        assert "does not call cheap models" in telemetry["notes"]

    def test_shadow_short_input_records_profit_guard(self):
        sc = SmartCompressor(
            main_model="mimo-v2.5-pro",
            api_key="sk-test-key",
            base_url="https://api.xiaomimimo.com/v1",
        )
        with patch.object(sc, '_call_compressor', side_effect=AssertionError("shadow must not call cheap model")):
            telemetry = sc.shadow_evaluate([{"role": "user", "content": "hi"}])
        assert telemetry["would_call_smart"] is False
        assert telemetry["would_fallback_reason"] == "short_input_profit_guard"
        assert telemetry["would_use_rule_only"] is True

    def test_shadow_protected_input_exposes_spans_and_policy(self):
        sc = self._make_configured()
        messages = [{
            "role": "user",
            "content": "Traceback Error at /app/data/project/main.py def parse_price(): 错误码 500，邮箱 support@unfaze.app，接口 https://api.example.com/v1/prices。" * 20,
        }]
        telemetry = sc.shadow_evaluate(messages)
        assert telemetry["policy_mode"] == "protected"
        assert telemetry["policy_target_ratio"] == 0.45
        assert telemetry["protected_span_count"] >= 4
        assert {"paths", "code_symbols", "numbers", "emails"}.issubset(set(telemetry["protected_span_kinds"]))
        assert "protected_policy" in telemetry["risk_flags"]
        assert "protected_spans" in telemetry["risk_flags"]

    def test_shadow_unknown_model_records_missing_route(self):
        sc = SmartCompressor(
            main_model="unknown-model",
            api_key="sk-test-key",
            base_url="https://api.example.com/v1",
            min_rule_tokens_for_smart=1,
        )
        telemetry = sc.shadow_evaluate([{"role": "user", "content": "请总结这段长内容" * 80}])
        assert telemetry["would_call_smart"] is False
        assert telemetry["would_fallback_reason"] == "missing_route_or_api_config"
        assert telemetry["route"] is None

    def test_shadow_context_guard_records_candidate_diagnostics(self):
        sc = self._make_configured()
        tiny_option = replace(sc.route.cheap_options[0], max_context=2)
        sc.route = replace(sc.route, cheap_options=(tiny_option,))
        sc.active_option = tiny_option
        telemetry = sc.shadow_evaluate([{"role": "user", "content": "这是一个明显超过两个token的长输入"}])
        assert telemetry["would_call_smart"] is False
        assert telemetry["would_fallback_reason"] == "context_guard"
        assert telemetry["candidate_diagnostics"][0]["blocked_by_context"] is True



def test_smart_compressor_model_probe_updates_route_and_metadata():
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "mimo-v2.5-pro"}, {"id": "mimo-v2-flash"}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            return FakeResponse()

    with patch("token_optimizer.core.model_probe.httpx.Client", FakeClient):
        sc = SmartCompressor(
            main_model="mimo-v2.5-pro",
            api_key="sk-test-key",
            base_url="https://platform.xiaomimimo.com/v1",
            min_rule_tokens_for_smart=1,
            enable_model_probe=True,
        )
    telemetry = sc.shadow_evaluate([{"role": "user", "content": "请总结这段很长的项目上下文" * 80}])
    assert telemetry["route"]["probe"]["enabled"] is True
    assert telemetry["route"]["probe"]["available"] is True
    assert telemetry["route"]["probe"]["source"] == "models_inventory"
    assert telemetry["selected_candidate"] == "mimo-v2-flash"
