"""ProductionOptimizer quickstart.

Run:
    export TOKEN_OPTIMIZER_API_KEY="sk-..."
    export TOKEN_OPTIMIZER_BASE_URL="https://api.xiaomimimo.com/v1"
    export TOKEN_OPTIMIZER_MODEL="mimo-v2.5-pro"
    python examples/production_quickstart.py
"""

from __future__ import annotations

import os

from token_optimizer import ProductionOptimizer, ProductionOptimizerConfig, RolloutGate


def main() -> None:
    api_key = os.getenv("TOKEN_OPTIMIZER_API_KEY", "")
    base_url = os.getenv("TOKEN_OPTIMIZER_BASE_URL", "https://api.xiaomimimo.com/v1")
    model = os.getenv("TOKEN_OPTIMIZER_MODEL", "mimo-v2.5-pro")
    rollout_mode = os.getenv("TOKEN_OPTIMIZER_ROLLOUT", "shadow")

    optimizer = ProductionOptimizer(ProductionOptimizerConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
        rollout=RolloutGate(mode=rollout_mode),
        enable_model_probe=os.getenv("TOKEN_OPTIMIZER_ENABLE_MODEL_PROBE", "1") == "1",
    ))

    response = optimizer.chat_completions_create(messages=[
        {"role": "system", "content": "你是一个严谨的技术助手。"},
        {"role": "user", "content": "请总结这段很长的项目上下文，并保留关键数字、路径和 API 参数。" * 80},
    ])

    print(response["choices"][0]["message"]["content"])
    print("\n--- optimization telemetry ---")
    print(response.get("_optimization", {}))


if __name__ == "__main__":
    main()
