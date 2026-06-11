#!/usr/bin/env python3
"""Run real-world token optimizer benchmark cases."""
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


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    result, info = run_token_optimizer_adaptive(case["messages"], runs=1)
    text = extract_context_text(result)
    passed = 0
    misses = []
    for group in case["qa_groups"]:
        found, _missing = keyword_recall("", text, group)
        if found:
            passed += 1
        else:
            misses.append("/".join(group))
    total = len(case["qa_groups"])
    return {
        "id": case["id"],
        "category": case["category"],
        "title": case.get("title", ""),
        "path": case.get("_path", ""),
        "tokens": info.get("tokens"),
        "keep_ratio": info.get("keep_ratio"),
        "method": info.get("method"),
        "qa_passed": passed,
        "qa_total": total,
        "qa_rate": passed / total if total else 0.0,
        "misses": misses,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-dir", type=Path, default=CASES_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "latest.json")
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

    print("Real-world Benchmark")
    print("=" * 96)
    print(f"{'ID':30} {'Category':22} {'QA':>9} {'Tokens':>8} {'Keep':>6} {'Method':>12}")
    print("-" * 96)
    for row in rows:
        qa = f"{row['qa_passed']}/{row['qa_total']}"
        keep = row["keep_ratio"]
        keep_s = f"{keep:.2f}" if isinstance(keep, (int, float)) else "-"
        print(
            f"{row['id'][:30]:30} {row['category'][:22]:22} "
            f"{qa:>9} {row['tokens']:>8} {keep_s:>6} {row['method']:>12}"
        )
        if row["misses"]:
            print(f"  MISS: {', '.join(row['misses'])}")

    print("\nSummary")
    print("-" * 96)
    print(f"Cases: {total_cases}")
    print(f"QA: {passed_qa}/{total_qa} = {summary['qa_rate']:.1%}")
    print(f"Perfect cases: {perfect_cases}/{total_cases}")
    print(f"Failed cases: {len(failed_cases)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nWrote: {args.output.relative_to(ROOT)}")
    return 0 if not failed_cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
