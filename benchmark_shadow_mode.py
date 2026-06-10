"""Shadow-mode telemetry benchmark for v5 SmartCompressor.

This does not call any cheap model and does not alter requests. It estimates what
v5 would do on the benchmark scenario mix and writes telemetry for rollout review.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark_l1_v5 import SCENARIOS
from token_optimizer.core.smart_compressor import SmartCompressor

ROOT = Path(__file__).parent
RESULTS_PATH = ROOT / "shadow_mode_results.json"
REPORT_PATH = ROOT / "shadow_mode_report.md"


def _scenario_messages(scenario: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in scenario:
        return scenario["messages"]
    return [
        {"role": "system", "content": scenario.get("system", "")},
        {"role": "user", "content": scenario.get("input", "")},
    ]


def run_shadow_mode() -> dict[str, Any]:
    compressor = SmartCompressor(
        main_model="mimo-v2.5-pro",
        api_key="shadow-key-present",
        base_url="https://api.xiaomimimo.com/v1",
        min_rule_tokens_for_smart=1,
    )
    rows = []
    policy_counter: Counter[str] = Counter()
    fallback_counter: Counter[str] = Counter()
    span_kind_counter: Counter[str] = Counter()
    totals = defaultdict(float)

    for scenario in SCENARIOS:
        telemetry = compressor.shadow_evaluate(_scenario_messages(scenario))
        row = {
            "scenario": scenario["name"],
            "would_call_smart": telemetry["would_call_smart"],
            "would_fallback_reason": telemetry["would_fallback_reason"],
            "original_tokens": telemetry["original_tokens"],
            "rule_tokens": telemetry["rule_tokens"],
            "estimated_smart_tokens": telemetry["estimated_smart_tokens"],
            "estimated_rule_cost": telemetry["estimated_rule_cost"],
            "estimated_smart_cost": telemetry["estimated_smart_cost"],
            "estimated_savings_pct": telemetry["estimated_savings_pct"],
            "estimated_raw_to_smart_savings_pct": telemetry.get("estimated_raw_to_smart_savings_pct"),
            "policy_mode": telemetry["policy_mode"],
            "policy_target_ratio": telemetry["policy_target_ratio"],
            "protected_span_count": telemetry["protected_span_count"],
            "protected_span_kinds": telemetry["protected_span_kinds"],
            "selected_candidate": telemetry["selected_candidate"],
            "risk_flags": telemetry["risk_flags"],
        }
        rows.append(row)
        policy_counter[row["policy_mode"]] += 1
        if row["would_fallback_reason"]:
            fallback_counter[row["would_fallback_reason"]] += 1
        for kind in row["protected_span_kinds"]:
            span_kind_counter[kind] += 1
        totals["original_tokens"] += row["original_tokens"] or 0
        totals["rule_tokens"] += row["rule_tokens"] or 0
        totals["estimated_smart_tokens"] += row["estimated_smart_tokens"] or row["rule_tokens"] or 0
        totals["estimated_rule_cost"] += row["estimated_rule_cost"] or 0
        totals["estimated_smart_cost"] += row["estimated_smart_cost"] or row["estimated_rule_cost"] or 0
        totals["smart_enabled"] += 1 if row["would_call_smart"] else 0

    rule_cost = totals["estimated_rule_cost"]
    smart_cost = totals["estimated_smart_cost"]
    summary = {
        "scenario_count": len(rows),
        "would_call_smart_count": int(totals["smart_enabled"]),
        "would_call_smart_rate": round(totals["smart_enabled"] / max(1, len(rows)), 4),
        "original_tokens": int(totals["original_tokens"]),
        "rule_tokens": int(totals["rule_tokens"]),
        "estimated_smart_tokens": int(totals["estimated_smart_tokens"]),
        "rule_to_smart_token_saved_pct": round((1 - totals["estimated_smart_tokens"] / max(1, totals["rule_tokens"])) * 100, 2),
        "estimated_rule_cost": round(rule_cost, 8),
        "estimated_smart_cost": round(smart_cost, 8),
        "estimated_incremental_cost_saved_pct": round((1 - smart_cost / rule_cost) * 100, 2) if rule_cost > 0 else 0.0,
        "policy_distribution": dict(policy_counter),
        "fallback_distribution": dict(fallback_counter),
        "protected_span_kind_distribution": dict(span_kind_counter),
    }
    return {"summary": summary, "rows": rows}


def write_report(results: dict[str, Any]) -> None:
    summary = results["summary"]
    lines = [
        "# Shadow Mode / Telemetry Report",
        "",
        "本报告用于上线前 dry-run：不改变真实请求、不调用廉价模型，只记录 v5 SmartCompressor 如果启用会发生什么。",
        "",
        "## Summary",
        "",
        f"- Scenarios: {summary['scenario_count']}",
        f"- Would call smart compression: {summary['would_call_smart_count']}/{summary['scenario_count']} ({summary['would_call_smart_rate']:.0%})",
        f"- Original tokens: {summary['original_tokens']}",
        f"- Rule tokens: {summary['rule_tokens']}",
        f"- Estimated smart tokens: {summary['estimated_smart_tokens']}",
        f"- Rule → smart token saved: {summary['rule_to_smart_token_saved_pct']}%",
        f"- Estimated rule cost: ${summary['estimated_rule_cost']:.8f}",
        f"- Estimated smart cost: ${summary['estimated_smart_cost']:.8f}",
        f"- Estimated incremental cost saved: {summary['estimated_incremental_cost_saved_pct']}%",
        "",
        "## Policy Distribution",
        "",
    ]
    for mode, count in summary["policy_distribution"].items():
        lines.append(f"- {mode}: {count}")
    lines += ["", "## Fallback Distribution", ""]
    if summary["fallback_distribution"]:
        for reason, count in summary["fallback_distribution"].items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines += ["", "## Scenario Rows", "", "| Scenario | Smart? | Policy | Rule tokens | Est. smart tokens | Est. savings | Protected spans | Candidate | Fallback |", "|---|---:|---|---:|---:|---:|---:|---|---|"]
    for row in results["rows"]:
        lines.append(
            "| {scenario} | {smart} | {policy} | {rule_tokens} | {smart_tokens} | {savings}% | {spans} | {candidate} | {fallback} |".format(
                scenario=row["scenario"],
                smart="yes" if row["would_call_smart"] else "no",
                policy=row["policy_mode"],
                rule_tokens=row["rule_tokens"],
                smart_tokens=row["estimated_smart_tokens"] if row["estimated_smart_tokens"] is not None else "-",
                savings=row["estimated_savings_pct"] if row["estimated_savings_pct"] is not None else "-",
                spans=row["protected_span_count"],
                candidate=row["selected_candidate"] or "-",
                fallback=row["would_fallback_reason"] or "-",
            )
        )
    lines += [
        "",
        "## Rollout Meaning",
        "",
        "- Shadow mode 可以先在线上旁路采样，不改变主链路输入。",
        "- 只记录收益、策略、protected spans、候选模型、fallback reason。",
        "- 当 telemetry 显示收益稳定且 fallback/保护分布合理，再逐步打开真实 smart compression。",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    results = run_shadow_mode()
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(results)
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {RESULTS_PATH.name} and {REPORT_PATH.name}")
