"""Dry-run ProductionOptimizer without external API calls.

This example monkeypatches the HTTP request method so users can inspect the final
optimization metadata locally before wiring real provider credentials.
"""

from __future__ import annotations

from token_optimizer import ProductionOptimizer, ProductionOptimizerConfig, RolloutGate


class LocalDemoOptimizer(ProductionOptimizer):
    def _sync_request(self, payload):  # type: ignore[override]
        return {
            "choices": [{"message": {"role": "assistant", "content": "demo response"}}],
            "usage": {"prompt_tokens": 128, "completion_tokens": 32},
            "_debug_payload_messages": payload["messages"],
        }


def main() -> None:
    optimizer = LocalDemoOptimizer(ProductionOptimizerConfig(
        model="mimo-v2.5-pro",
        api_key="sk-demo",
        base_url="https://api.xiaomimimo.com/v1",
        rollout=RolloutGate(mode="auto", min_estimated_savings_pct=20),
        enable_model_probe=False,
        smart_min_rule_tokens=1,
    ))
    response = optimizer.chat_completions_create(messages=[
        {"role": "user", "content": "Traceback Error at /app/data/project/main.py def parse_price(): 错误码 500，接口 https://api.example.com/v1/prices。" * 30},
    ])
    opt = response["_optimization"]
    print("final_path:", opt["final_path"])
    print("rollout_gate:", opt["rollout_gate"])
    print("token_saved_pct_est:", opt["token_saved_pct_est"])
    print("protected_span_count:", opt["shadow"].get("protected_span_count"))


if __name__ == "__main__":
    main()
