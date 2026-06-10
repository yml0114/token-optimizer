"""Local smoke test for the production optimizer pipeline.

No external API call is made. The test verifies the packaged SDK surface and the
optimization metadata contract.
"""

from __future__ import annotations

from token_optimizer import ProductionOptimizer, ProductionOptimizerConfig, RolloutGate


class SmokeOptimizer(ProductionOptimizer):
    def _sync_request(self, payload):  # type: ignore[override]
        return {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 64, "completion_tokens": 16},
        }


def main() -> None:
    optimizer = SmokeOptimizer(ProductionOptimizerConfig(
        model="mimo-v2.5-pro",
        api_key="sk-smoke",
        base_url="https://api.xiaomimimo.com/v1",
        rollout=RolloutGate(mode="auto", min_estimated_savings_pct=20),
        enable_model_probe=False,
        smart_min_rule_tokens=1,
    ))
    response = optimizer.chat_completions_create(messages=[
        {"role": "user", "content": "请总结这段很长的上下文，保留 /app/data/a.py 和 500 错误码。" * 60},
    ])
    opt = response.get("_optimization", {})
    assert response["choices"][0]["message"]["content"] == "ok"
    assert opt.get("production_optimizer") is True
    assert opt.get("shadow", {}).get("mode") == "shadow"
    assert "rollout_gate" in opt
    assert "l1_smart_compression" in opt
    assert "token_saved_pct_est" in opt
    print("production smoke ok")
    print({
        "final_path": opt.get("final_path"),
        "rollout_gate": opt.get("rollout_gate"),
        "token_saved_pct_est": opt.get("token_saved_pct_est"),
    })


if __name__ == "__main__":
    main()
