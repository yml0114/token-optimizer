"""Regression benchmark for L1 v5 compression fidelity guard.

This benchmark is deterministic and does not call external APIs. It checks that
lossy compressor outputs are rejected while safe outputs pass.
"""

from __future__ import annotations

from unittest.mock import patch

from token_optimizer.core.smart_compressor import SmartCompressor, estimate_tokens_from_messages


CASES = [
    {
        "name": "code_path_number_url",
        "messages": [{"role": "user", "content": ("修复 /app/data/project/main.py 的 parse_price()，错误码 500，金额 ¥19.9，接口 https://api.example.com/v1/prices。" * 18)}],
        "safe": [{"role": "user", "content": "修复 /app/data/project/main.py 的 parse_price()；保留错误码 500、金额 ¥19.9、接口 https://api.example.com/v1/prices。"}],
        "lossy": [{"role": "user", "content": "修复价格解析函数。"}],
    },
    {
        "name": "json_api_params",
        "messages": [{"role": "user", "content": ('请求体 {"user_id": 42, "plan": "pro"}，必须保留 endpoint /v1/billing/checkout 和 status=409。' * 20)}],
        "safe": [{"role": "user", "content": '保留 {"user_id": 42, "plan": "pro"}、/v1/billing/checkout、status=409。'}],
        "lossy": [{"role": "user", "content": "保留计费请求参数。"}],
    },
    {
        "name": "low_risk_history",
        "messages": [{"role": "assistant", "content": ("好的，谢谢补充。这个历史背景大概总结一下即可。" * 40)} , {"role": "user", "content": "总结历史背景"}],
        "safe": [{"role": "user", "content": "总结历史背景"}],
        "lossy": [{"role": "user", "content": "处理一下"}],
    },
]


def run_case(sc: SmartCompressor, messages, output):
    with patch.object(sc, "_call_compressor", return_value=output):
        result, meta = sc.compress(messages)
    return result, meta


def main() -> None:
    sc = SmartCompressor(
        main_model="mimo-v2.5-pro",
        api_key="sk-test-key",
        base_url="https://api.xiaomimimo.com/v1",
        min_rule_tokens_for_smart=1,
    )
    passed = 0
    rejected = 0
    print("L1 v5 Fidelity Regression Benchmark")
    print("=" * 72)
    for case in CASES:
        _, safe_meta = run_case(sc, case["messages"], case["safe"])
        _, lossy_meta = run_case(sc, case["messages"], case["lossy"])
        original_tokens = estimate_tokens_from_messages(case["messages"])
        safe_tokens = estimate_tokens_from_messages(case["safe"])
        safe_ok = safe_meta["mode"] == "smart" and safe_meta["fidelity_guard"]["passed"]
        lossy_rejected = lossy_meta["mode"] == "rule_only_fidelity_guard"
        passed += int(safe_ok)
        rejected += int(lossy_rejected)
        print(f"{case['name']:<24} original={original_tokens:<5} safe={safe_tokens:<4} "
              f"safe_ok={safe_ok!s:<5} lossy_rejected={lossy_rejected!s:<5} "
              f"safe_score={safe_meta.get('fidelity_guard', {}).get('score')} "
              f"lossy_score={lossy_meta.get('fidelity_guard', {}).get('score')}")
    total = len(CASES)
    print("-" * 72)
    print(f"Safe pass rate:      {passed}/{total}")
    print(f"Lossy reject rate:   {rejected}/{total}")
    if passed != total or rejected != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
