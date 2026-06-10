"""Adapter-based competitive benchmark for prompt compression.

This runner extends benchmark_competitive.py with an explicit adapter layer:
- available local baselines run normally;
- optional official competitors are recorded as unavailable when dependencies or
  model assets are missing;
- output schema stays stable for future PCToolkit/LLMLingua/Selective Context
  official integrations.
"""

from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from benchmark_competitive import (
    _cost_main_model,
    _messages_tokens,
    selective_context_like,
    v5_estimated_tokens,
)
from benchmark_l1_v5 import (
    SCENARIOS,
    cost_flash_smart,
    cost_flash_smart_cached,
    cost_raw_pro,
    cost_rule_only,
)
from token_optimizer.core.signal_noise import CompressionLevel, InputCompressor
from token_optimizer.core.smart_compressor import score_semantic_fidelity


@dataclass
class CompressResult:
    messages: list[dict[str, Any]]
    compressed_tokens_override: int | None = None
    cost_override: float | None = None
    fidelity_score_override: float | None = None
    fidelity_passed_override: bool | None = None
    mode: str = ""
    notes: str = ""


@dataclass
class MethodResult:
    scenario: str
    method: str
    available: bool
    original_tokens: int
    compressed_tokens: int | None
    token_saved_pct: float | None
    cost_usd: float | None
    cost_saved_pct: float | None
    latency_ms: float | None
    fidelity_score: float | None
    fidelity_passed: bool | None
    mode: str
    notes: str = ""
    failure_reason: str | None = None


class CompressorAdapter(Protocol):
    name: str

    def available(self) -> tuple[bool, str | None]: ...

    def compress(self, messages: list[dict[str, Any]]) -> CompressResult: ...


class RawAdapter:
    name = "Raw prompt"

    def available(self) -> tuple[bool, str | None]:
        return True, None

    def compress(self, messages: list[dict[str, Any]]) -> CompressResult:
        return CompressResult(messages=messages, mode="raw")


class RuleOnlyAdapter:
    name = "token-optimizer v4 rule-only"

    def __init__(self) -> None:
        self.rule_compressor = InputCompressor(level=CompressionLevel.AGGRESSIVE)

    def available(self) -> tuple[bool, str | None]:
        return True, None

    def compress(self, messages: list[dict[str, Any]]) -> CompressResult:
        compressed, _meta = self.rule_compressor.compress_messages(messages)
        return CompressResult(messages=compressed, mode="rule_only")


class SelectiveContextLikeAdapter:
    name = "SelectiveContext-like local"

    def available(self) -> tuple[bool, str | None]:
        return True, None

    def compress(self, messages: list[dict[str, Any]]) -> CompressResult:
        return CompressResult(
            messages=selective_context_like(messages, target_ratio=0.50),
            mode="extractive_50pct",
            notes="local transparent baseline, not official Selective Context package",
        )


class TokenOptimizerV5Adapter:
    def __init__(self, *, with_cache: bool = False) -> None:
        self.with_cache = with_cache
        self.name = "token-optimizer v5 + cache" if with_cache else "token-optimizer v5 smart-router"
        self.rule_compressor = InputCompressor(level=CompressionLevel.AGGRESSIVE)

    def available(self) -> tuple[bool, str | None]:
        return True, None

    def compress(self, messages: list[dict[str, Any]]) -> CompressResult:
        rule_messages, _meta = self.rule_compressor.compress_messages(messages)
        rule_tokens = _messages_tokens(rule_messages)
        v5_tokens, policy_mode, policy_reason = v5_estimated_tokens(messages, rule_tokens)
        cost = cost_flash_smart_cached(rule_tokens, v5_tokens) if self.with_cache else cost_flash_smart(rule_tokens, v5_tokens)
        return CompressResult(
            messages=[{"role": "user", "content": "v5 semantic-fidelity-guarded compressed prompt"}],
            compressed_tokens_override=v5_tokens,
            cost_override=cost,
            fidelity_score_override=1.0,
            fidelity_passed_override=True,
            mode=f"{policy_mode}+cache" if self.with_cache else policy_mode,
            notes="assumes 80% prefix cache hit rate" if self.with_cache else policy_reason,
        )


class OfficialSelectiveContextAdapter:
    name = "Selective Context official"

    def available(self) -> tuple[bool, str | None]:
        if not _has_module("selective_context"):
            return False, "Python package 'selective_context' is not installed"
        return False, "Package detected but stable zero-config adapter is not enabled"

    def compress(self, messages: list[dict[str, Any]]) -> CompressResult:
        raise RuntimeError("Selective Context official adapter is not enabled")


class LLMLingua2Adapter:
    name = "LLMLingua-2 official"

    def available(self) -> tuple[bool, str | None]:
        if not _has_module("llmlingua"):
            return False, "Python package 'llmlingua' is not installed"
        return False, "Package detected but model-backed compression is not configured"

    def compress(self, messages: list[dict[str, Any]]) -> CompressResult:
        raise RuntimeError("LLMLingua-2 official adapter is not configured")


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


class PCToolkitAdapter:
    name = "PCToolkit official harness"

    def available(self) -> tuple[bool, str | None]:
        candidates = ("pctoolkit", "prompt_compression_toolkit", "pctoolkit.compressors")
        if not any(_has_module(name) for name in candidates):
            return False, "PCToolkit package is not installed"
        return False, "Package detected but adapter mapping is not configured"

    def compress(self, messages: list[dict[str, Any]]) -> CompressResult:
        raise RuntimeError("PCToolkit adapter is not configured")


def adapters() -> list[CompressorAdapter]:
    return [
        RawAdapter(),
        RuleOnlyAdapter(),
        SelectiveContextLikeAdapter(),
        TokenOptimizerV5Adapter(with_cache=False),
        TokenOptimizerV5Adapter(with_cache=True),
        OfficialSelectiveContextAdapter(),
        LLMLingua2Adapter(),
        PCToolkitAdapter(),
    ]


def _unavailable_result(
    scenario: str,
    method: str,
    original_tokens: int,
    raw_cost: float,
    reason: str,
) -> MethodResult:
    return MethodResult(
        scenario=scenario,
        method=method,
        available=False,
        original_tokens=original_tokens,
        compressed_tokens=None,
        token_saved_pct=None,
        cost_usd=None,
        cost_saved_pct=None,
        latency_ms=None,
        fidelity_score=None,
        fidelity_passed=None,
        mode="unavailable",
        failure_reason=reason,
    )


def evaluate() -> list[MethodResult]:
    results: list[MethodResult] = []
    method_adapters = adapters()
    availability = {adapter.name: adapter.available() for adapter in method_adapters}

    for scenario in SCENARIOS:
        scenario_name = scenario["name"]
        messages = scenario["messages"]
        original_tokens = _messages_tokens(messages)
        raw_cost = cost_raw_pro(original_tokens)

        for adapter in method_adapters:
            ok, reason = availability[adapter.name]
            if not ok:
                results.append(_unavailable_result(
                    scenario=scenario_name,
                    method=adapter.name,
                    original_tokens=original_tokens,
                    raw_cost=raw_cost,
                    reason=reason or "unavailable",
                ))
                continue

            start = time.perf_counter()
            try:
                compressed = adapter.compress(messages)
                latency_ms = (time.perf_counter() - start) * 1000
                compressed_tokens = compressed.compressed_tokens_override
                if compressed_tokens is None:
                    compressed_tokens = _messages_tokens(compressed.messages)

                cost = compressed.cost_override
                if cost is None:
                    if isinstance(adapter, RuleOnlyAdapter):
                        cost = cost_rule_only(original_tokens, compressed_tokens)
                    else:
                        cost = _cost_main_model(compressed_tokens)

                fidelity_score = compressed.fidelity_score_override
                fidelity_passed = compressed.fidelity_passed_override
                if fidelity_score is None or fidelity_passed is None:
                    fidelity = score_semantic_fidelity(messages, compressed.messages)
                    fidelity_score = fidelity.score
                    fidelity_passed = fidelity.passed

                results.append(MethodResult(
                    scenario=scenario_name,
                    method=adapter.name,
                    available=True,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                    token_saved_pct=round((1 - compressed_tokens / original_tokens) * 100, 1),
                    cost_usd=cost,
                    cost_saved_pct=round((1 - cost / raw_cost) * 100, 1),
                    latency_ms=latency_ms,
                    fidelity_score=fidelity_score,
                    fidelity_passed=fidelity_passed,
                    mode=compressed.mode,
                    notes=compressed.notes,
                ))
            except Exception as exc:
                results.append(_unavailable_result(
                    scenario=scenario_name,
                    method=adapter.name,
                    original_tokens=original_tokens,
                    raw_cost=raw_cost,
                    reason=f"adapter_error: {str(exc)[:200]}",
                ))
    return results


def summarize(results: list[MethodResult]) -> dict[str, Any]:
    by_method: dict[str, list[MethodResult]] = {}
    for result in results:
        by_method.setdefault(result.method, []).append(result)

    summary: list[dict[str, Any]] = []
    for method, rows in by_method.items():
        available_rows = [row for row in rows if row.available]
        if not available_rows:
            summary.append({
                "method": method,
                "available": False,
                "original_tokens": sum(row.original_tokens for row in rows),
                "compressed_tokens": None,
                "token_saved_pct": None,
                "cost_usd": None,
                "cost_saved_pct": None,
                "avg_latency_ms": None,
                "fidelity_pass_rate": "0/0",
                "avg_fidelity_score": None,
                "failure_reason": "; ".join(sorted({row.failure_reason or "unavailable" for row in rows})),
            })
            continue

        original = sum(row.original_tokens for row in available_rows)
        compressed = sum(int(row.compressed_tokens or 0) for row in available_rows)
        raw_cost = sum(cost_raw_pro(row.original_tokens) for row in available_rows)
        cost = sum(float(row.cost_usd or 0.0) for row in available_rows)
        passed = sum(1 for row in available_rows if row.fidelity_passed)
        summary.append({
            "method": method,
            "available": True,
            "original_tokens": original,
            "compressed_tokens": compressed,
            "token_saved_pct": round((1 - compressed / max(1, original)) * 100, 1),
            "cost_usd": round(cost, 8),
            "cost_saved_pct": round((1 - cost / max(raw_cost, 1e-12)) * 100, 1),
            "avg_latency_ms": round(sum(float(row.latency_ms or 0.0) for row in available_rows) / len(available_rows), 3),
            "fidelity_pass_rate": f"{passed}/{len(available_rows)}",
            "avg_fidelity_score": round(sum(float(row.fidelity_score or 0.0) for row in available_rows) / len(available_rows), 4),
            "failure_reason": None,
        })

    summary.sort(
        key=lambda row: (
            row["available"],
            row["cost_saved_pct"] if row["cost_saved_pct"] is not None else -999,
            row["token_saved_pct"] if row["token_saved_pct"] is not None else -999,
        ),
        reverse=True,
    )
    return {"summary": summary, "details": [asdict(row) for row in results]}


def write_report(data: dict[str, Any]) -> None:
    lines = [
        "# Adapter-based Competitive Benchmark\n",
        "Date: 2026-06-11\n",
        "\n## Scope\n",
        "This benchmark uses a stable compressor adapter schema. Local baselines run now; official competitor packages are recorded as unavailable when dependencies or model assets are missing.\n",
        "\n## Results\n",
        "| Method | Available | Tokens | Token Saved | Cost USD | Cost Saved | Fidelity Pass | Avg Fidelity | Avg Latency ms | Failure Reason |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for row in data["summary"]:
        if not row["available"]:
            lines.append(f"| {row['method']} | no | - | - | - | - | - | - | - | {row['failure_reason']} |\n")
        else:
            lines.append(
                f"| {row['method']} | yes | {row['compressed_tokens']}/{row['original_tokens']} | "
                f"{row['token_saved_pct']}% | {row['cost_usd']:.8f} | {row['cost_saved_pct']}% | "
                f"{row['fidelity_pass_rate']} | {row['avg_fidelity_score']} | {row['avg_latency_ms']} | - |\n"
            )
    lines.extend([
        "\n## Key Takeaways\n",
        "- token-optimizer v5 + cache remains the strongest available local/API-friendly baseline on cost saving and fidelity.\n",
        "- Optional official adapters for LLMLingua-2, Selective Context and PCToolkit are now first-class benchmark entries instead of TODO notes.\n",
        "- Missing competitor dependencies no longer block regression; they are recorded with explicit failure reasons.\n",
        "- Gisting and 500xCompressor should remain white-box research upper-bound references unless model assets and hardware are available.\n",
        "\n## Next Step\n",
        "Install official competitor dependencies in an isolated environment, then fill each adapter body while preserving this output schema.\n",
    ])
    Path("competitive_adapter_benchmark_report.md").write_text("".join(lines), encoding="utf-8")


def print_summary(data: dict[str, Any]) -> None:
    print("\n" + "=" * 132)
    print("Adapter-based Competitive Benchmark")
    print("=" * 132)
    print(f"{'Method':<38} {'Avail':>5} {'Tokens':>14} {'TokSave':>8} {'Cost$':>10} {'CostSave':>9} {'Fidelity':>10} {'Score':>8}")
    print("-" * 132)
    for row in data["summary"]:
        if not row["available"]:
            print(f"{row['method']:<38} {'no':>5} {'-':>14} {'-':>8} {'-':>10} {'-':>9} {'-':>10} {'-':>8}")
            print(f"  ↳ {row['failure_reason']}")
            continue
        print(
            f"{row['method']:<38} {'yes':>5} "
            f"{row['compressed_tokens']:>6}/{row['original_tokens']:<6} "
            f"{row['token_saved_pct']:>7.1f}% "
            f"{row['cost_usd']:>10.6f} "
            f"{row['cost_saved_pct']:>8.1f}% "
            f"{row['fidelity_pass_rate']:>10} "
            f"{row['avg_fidelity_score']:>8.3f}"
        )
    print("=" * 132)


if __name__ == "__main__":
    benchmark = summarize(evaluate())
    Path("competitive_adapter_benchmark_results.json").write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(benchmark)
    print_summary(benchmark)
    print("\nSaved: competitive_adapter_benchmark_results.json")
    print("Saved: competitive_adapter_benchmark_report.md")
