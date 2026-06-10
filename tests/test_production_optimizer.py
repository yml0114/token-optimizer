"""Tests for the production-ready optimizer wrapper."""

from unittest.mock import patch

from token_optimizer.production import ProductionOptimizer, ProductionOptimizerConfig, RolloutGate


class FakeResponse:
    def __init__(self, payload=None):
        self.payload = payload or {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {"prompt_tokens": 100, "completion_tokens": 20}}

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeClient:
    last_payload = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, json=None):
        FakeClient.last_payload = json
        return FakeResponse()


class TestRolloutGate:
    def test_shadow_mode_never_enables_real_smart(self):
        gate = RolloutGate(mode="shadow")
        assert gate.decide({"would_call_smart": True})["enabled"] is False

    def test_auto_gate_requires_savings(self):
        gate = RolloutGate(mode="auto", min_estimated_savings_pct=20)
        decision = gate.decide({"would_call_smart": True, "estimated_savings_pct": 5, "protected_span_count": 0})
        assert decision["enabled"] is False
        assert decision["reason"] == "estimated_savings_below_threshold"

    def test_auto_gate_passes_profitable_shadow(self):
        gate = RolloutGate(mode="auto", min_estimated_savings_pct=20)
        decision = gate.decide({"would_call_smart": True, "estimated_savings_pct": 50, "protected_span_count": 0, "route": {}})
        assert decision["enabled"] is True


class TestProductionOptimizer:
    def _optimizer(self, rollout=None):
        return ProductionOptimizer(ProductionOptimizerConfig(
            model="mimo-v2.5-pro",
            api_key="sk-test",
            base_url="https://api.xiaomimimo.com/v1",
            enable_model_probe=False,
            smart_min_rule_tokens=1,
            rollout=rollout or RolloutGate(mode="shadow"),
        ))

    def test_shadow_rollout_sends_rule_optimized_payload_and_metadata(self):
        optimizer = self._optimizer(RolloutGate(mode="shadow"))
        messages = [{"role": "user", "content": "请总结这段很长的项目上下文" * 80}]
        with patch("token_optimizer.production.httpx.Client", FakeClient):
            response = optimizer.chat_completions_create(messages=messages)
        assert response["_optimization"]["production_optimizer"] is True
        assert response["_optimization"]["rollout_gate"]["enabled"] is False
        assert response["_optimization"]["shadow"]["mode"] == "shadow"
        assert FakeClient.last_payload["model"] == "mimo-v2.5-pro"
        assert "_optimization" in response

    def test_on_rollout_invokes_smart_compressor_when_gate_forced(self):
        optimizer = self._optimizer(RolloutGate(mode="on"))
        messages = [{"role": "user", "content": "请总结这段很长的项目上下文" * 80}]
        with patch.object(optimizer.smart, "_call_compressor", return_value=[{"role": "user", "content": "压缩摘要"}]):
            with patch("token_optimizer.production.httpx.Client", FakeClient):
                response = optimizer.chat_completions_create(messages=messages)
        assert response["_optimization"]["rollout_gate"]["enabled"] is True
        assert response["_optimization"]["l1_smart_compression"]["mode"] in {"smart", "rule_only_profit_guard", "rule_only_fidelity_guard", "rule_only_fallback"}

    def test_compressor_exception_never_breaks_main_request(self):
        optimizer = self._optimizer(RolloutGate(mode="on"))
        messages = [{"role": "user", "content": "请总结这段很长的项目上下文" * 80}]
        with patch.object(optimizer.smart, "compress", side_effect=RuntimeError("boom")):
            with patch("token_optimizer.production.httpx.Client", FakeClient):
                response = optimizer.chat_completions_create(messages=messages)
        assert response["choices"][0]["message"]["content"] == "ok"
        assert response["_optimization"]["l1_smart_compression"]["mode"] == "safe_passthrough_repair"
        assert response["_optimization"]["errors"]
