#!/usr/bin/env python3
"""Run adversarial token optimizer benchmark cases."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_benchmark import extract_context_text, keyword_recall, run_token_optimizer_adaptive

CASES_DIR = Path(__file__).resolve().parent / "cases"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_cases(cases_dir: Path) -> list[dict[str, Any]]:
    cases = []
    for path in sorted(cases_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            case = json.load(f)
        case["_path"] = str(path.relative_to(ROOT))
        cases.append(case)
    return cases


def _assertion_label(assertion: dict[str, Any]) -> str:
    if assertion.get("label"):
        return str(assertion["label"])
    values = assertion.get("values") or assertion.get("current") or assertion.get("value") or []
    if isinstance(values, str):
        return values
    return "/".join(str(v) for v in values)


def _classify_miss(assertion: dict[str, Any], missing: list[str]) -> str:
    assertion_type = assertion.get("type", "any_of")
    values = assertion.get("values") or assertion.get("current") or assertion.get("value") or []
    if isinstance(values, str):
        values = [values]
    if assertion_type not in {"present", "any_of", "all_of", "latest_value"}:
        return "assertion_gap"
    if assertion_type == "any_of" and len(values) > 1:
        return "alias_gap"
    if assertion_type == "latest_value":
        return "true_loss"
    return "true_loss"


def _evaluate_assertion(text: str, assertion: dict[str, Any]) -> tuple[bool, list[str], str]:
    assertion_type = assertion.get("type", "any_of")

    if assertion_type == "present":
        value = assertion.get("value") or assertion.get("values", [])
        values = [value] if isinstance(value, str) else list(value)
        found, missing = keyword_recall("", text, values)
    elif assertion_type == "any_of":
        values = assertion.get("values", [])
        found, missing = keyword_recall("", text, list(values))
    elif assertion_type == "all_of":
        missing = []
        for value in assertion.get("values", []):
            ok, _ = keyword_recall("", text, [value])
            if not ok:
                missing.append(value)
        found = not missing
    elif assertion_type == "latest_value":
        current = assertion.get("current", [])
        current_values = [current] if isinstance(current, str) else list(current)
        found, missing = keyword_recall("", text, current_values)
    else:
        found = False
        missing = [f"unsupported assertion type: {assertion_type}"]

    return found, missing, _classify_miss(assertion, missing)


def _qa_groups_to_assertions(case: dict[str, Any]) -> list[dict[str, Any]]:
    if "qa_assertions" in case:
        return list(case["qa_assertions"])
    return [
        {"type": "any_of", "label": "/".join(group), "values": group}
        for group in case.get("qa_groups", [])
    ]


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    result, info = run_token_optimizer_adaptive(case["messages"], runs=1)
    text = extract_context_text(result)
    passed = 0
    misses = []
    miss_details = []
    assertions = _qa_groups_to_assertions(case)
    for assertion in assertions:
        found, missing, classification = _evaluate_assertion(text, assertion)
        label = _assertion_label(assertion)
        if found:
            passed += 1
        else:
            misses.append(label)
            miss_details.append({
                "label": label,
                "type": assertion.get("type", "any_of"),
                "missing": missing,
                "classification": classification,
            })
    total = len(assertions)
    miss_summary = {
        "alias_gap": sum(1 for miss in miss_details if miss["classification"] == "alias_gap"),
        "true_loss": sum(1 for miss in miss_details if miss["classification"] == "true_loss"),
        "assertion_gap": sum(1 for miss in miss_details if miss["classification"] == "assertion_gap"),
    }
    return {
        "id": case["id"],
        "category": case["category"],
        "title": case.get("title", ""),
        "attack": case.get("attack", ""),
        "path": case.get("_path", ""),
        "tokens": info.get("tokens"),
        "keep_ratio": info.get("keep_ratio"),
        "method": info.get("method"),
        "qa_passed": passed,
        "qa_total": total,
        "qa_rate": passed / total if total else 0.0,
        "misses": misses,
        "miss_details": miss_details,
        "miss_summary": miss_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-dir", type=Path, default=CASES_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "latest.json")
    parser.add_argument("--min-rate", type=float, default=0.0, help="Optional minimum QA rate gate, e.g. 0.85")
    args = parser.parse_args()

    cases = load_cases(args.cases_dir)
    rows = [evaluate_case(case) for case in cases]

    total_cases = len(rows)
    total_qa = sum(row["qa_total"] for row in rows)
    passed_qa = sum(row["qa_passed"] for row in rows)
    perfect_cases = sum(1 for row in rows if row["qa_passed"] == row["qa_total"])
    failed_cases = [row for row in rows if row["misses"]]
    summary = {
        "total_cases": total_cases,
        "total_qa": total_qa,
        "passed_qa": passed_qa,
        "qa_rate": passed_qa / total_qa if total_qa else 0.0,
        "perfect_cases": perfect_cases,
        "failed_cases": failed_cases,
        "rows": rows,
    }

    print("Adversarial Benchmark")
    print("=" * 108)
    print(f"{'ID':30} {'Category':20} {'QA':>9} {'Tokens':>8} {'Keep':>6} {'Method':>12} {'Attack'}")
    print("-" * 108)
    for row in rows:
        qa = f"{row['qa_passed']}/{row['qa_total']}"
        keep = row["keep_ratio"]
        keep_s = f"{keep:.2f}" if isinstance(keep, (int, float)) else "-"
        print(
            f"{row['id'][:30]:30} {row['category'][:20]:20} "
            f"{qa:>9} {row['tokens']:>8} {keep_s:>6} {row['method']:>12} {row['attack'][:24]}"
        )
        if row["misses"]:
            print(f"  MISS: {', '.join(row['misses'])}")

    print("\nSummary")
    print("-" * 108)
    print(f"Cases: {total_cases}")
    print(f"QA: {passed_qa}/{total_qa} = {summary['qa_rate']:.1%}")
    print(f"Perfect cases: {perfect_cases}/{total_cases}")
    print(f"Failed cases: {len(failed_cases)}")
    if args.min_rate:
        print(f"Min-rate gate: {args.min_rate:.1%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nWrote: {args.output.relative_to(ROOT)}")

    if args.min_rate and summary["qa_rate"] < args.min_rate:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
