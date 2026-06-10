"""Benchmark L1 v2: Fragment-level compression on real conversation data.

Tests v2 improvements:
  - Inline filler stripping
  - Tool output bulk metadata removal
  - History compression (old turn summarization)
  - Combined pipeline

Run: python benchmark_l1_v2.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from token_optimizer.core.signal_noise import (
    InputCompressor,
    CompressionLevel,
    HistoryCompressor,
    ToolOutputCleaner,
)


def measure_tokens(text: str) -> int:
    return max(1, len(text) // 3)


SYSTEM_PROMPT = """You are a senior Python developer. Write clean, well-documented code.
Always use type hints. Prefer dataclasses over dicts for structured data.
Error handling: use specific exceptions, not bare except.
Testing: write tests for all public functions.
Code style: PEP 8 compliant, 4-space indentation, max line length 88.
When asked to explain, be concise — 3 sentences max."""


CONVERSATIONS = [
    {
        "name": "Coding request with Chinese fillers",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "请帮我写一个函数，如果可以的话用Python。我想实现一个快速排序算法，麻烦你了。"},
            {"role": "assistant", "content": "def quicksort(arr: list[int]) -> list[int]:\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"},
            {"role": "user", "content": "好的，谢谢！那么如果可以的话，能加一个原地排序的版本吗？不用创建新数组那种。"},
        ],
    },
    {
        "name": "Tool output with HTTP/trace noise",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "帮我查一下GitHub上token-optimizer项目的最新commit"},
            {"role": "tool", "content": 'HTTP/1.1 200 OK\nContent-Type: application/json\nX-RateLimit-Remaining: 59\nX-RateLimit-Reset: 1718064000\nrequest_id: abc123def456\ntrace_id: xyz789\nlatency: 45ms\n{"sha":"a1b2c3d","message":"feat: add L1 compression","date":"2026-06-10"}'},
            {"role": "user", "content": "看起来不错，请问这个commit具体做了什么改动？如果可以的话详细说一下。"},
        ],
    },
    {
        "name": "Error debugging with stack trace",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "运行出错了，帮我看看。不好意思，之前忘了检查类型。"},
            {"role": "assistant", "content": 'File "main.py", line 42\n    result = process_data(raw_input)\nFile "utils.py", line 15\n    return {k: v.strip() for k, v in data.items()}\nAttributeError: \'NoneType\' object has no attribute \'strip\''},
            {"role": "user", "content": "请帮我修复这个错误，麻烦加一个None检查。如果可以的话也加个单元测试。"},
        ],
    },
    {
        "name": "Long multi-turn with system tags and fillers",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<system_hint>Internal context: user is working on token-optimizer project</system_hint>\n请帮我写一个缓存管理器。"},
            {"role": "assistant", "content": "class CacheManager:\n    def __init__(self, max_size: int = 1000):\n        self._cache: dict[str, Any] = {}\n        self._max_size = max_size\n    \n    def get(self, key: str) -> Any | None:\n        return self._cache.get(key)\n    \n    def put(self, key: str, value: Any) -> None:\n        if len(self._cache) >= self._max_size:\n            self._evict_oldest()\n        self._cache[key] = value"},
            {"role": "user", "content": "好的谢谢！对了另外，能加一个LRU淘汰策略吗？嗯还有统计命中率的功能。"},
            {"role": "assistant", "content": "from collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, max_size: int = 1000):\n        self._cache = OrderedDict()\n        self._max_size = max_size\n        self._hits = 0\n        self._misses = 0\n    \n    def get(self, key: str) -> Any | None:\n        if key in self._cache:\n            self._hits += 1\n            self._cache.move_to_end(key)\n            return self._cache[key]\n        self._misses += 1\n        return None\n    \n    @property\n    def hit_rate(self) -> float:\n        total = self._hits + self._misses\n        return self._hits / total if total > 0 else 0.0"},
            {"role": "user", "content": "确实很好！请帮我写一个单元测试来验证LRU淘汰逻辑。"},
        ],
    },
    {
        "name": "English coding with API noise",
        "messages": [
            {"role": "system", "content": "You are a TypeScript expert. Write production-grade code."},
            {"role": "user", "content": "Please help me write a debounce function in TypeScript. If you don't mind, make it generic so it works with any return type."},
            {"role": "assistant", "content": "function debounce<T extends (...args: any[]) => any>(fn: T, delay: number): (...args: Parameters<T>) => void {\n  let timer: ReturnType<typeof setTimeout>;\n  return (...args: Parameters<T>) => {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn(...args), delay);\n  };\n}"},
            {"role": "user", "content": "Thanks! Also, when you get a chance, could you add leading edge support too? And maybe a cancel method."},
        ],
    },
    # ── NEW: Heavy noise scenario ──
    {
        "name": "Tool-heavy with metadata blocks",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "帮我查3个API的数据"},
            {"role": "tool", "content": "HTTP/1.1 200 OK\nContent-Type: application/json\nX-Request-Id: r1\nX-Trace: t1\nlatency: 30ms\n{\"users\": [{\"id\": 1}, {\"id\": 2}, {\"id\": 3}], \"total\": 3}"},
            {"role": "tool", "content": "HTTP/1.1 200 OK\nContent-Type: application/json\nX-Request-Id: r2\nX-Trace: t2\nlatency: 45ms\n{\"repos\": [{\"name\": \"token-optimizer\", \"stars\": 12}]}"},
            {"role": "tool", "content": "HTTP/1.1 200 OK\nContent-Type: application/json\nX-Request-Id: r3\nX-Trace: t3\nlatency: 20ms\n{\"commits\": [{\"sha\": \"abc123\", \"msg\": \"feat: init\"}]}"},
            {"role": "assistant", "content": "Here are the results from all 3 APIs:\n\n1. Users: 3 users found\n2. Repos: token-optimizer (12 stars)\n3. Commits: abc123 - feat: init"},
            {"role": "user", "content": "好的谢谢！请问这些数据中，如果可以的话，能帮我分析一下哪个repo的star最多吗？"},
        ],
    },
]


def run_benchmark():
    print("=" * 70)
    print("L1 v2 Benchmark — Fragment-Level Compression + History + Tool Cleaning")
    print("=" * 70)

    levels = [CompressionLevel.SAFE, CompressionLevel.MODERATE, CompressionLevel.AGGRESSIVE]
    all_results = []

    for level in levels:
        print(f"\n{'─' * 70}")
        print(f"Compression Level: {level.value.upper()}")
        print(f"{'─' * 70}")

        compressor = InputCompressor(level=level)
        level_results = []

        for conv in CONVERSATIONS:
            name = conv["name"]
            msgs = conv["messages"]
            system_text = msgs[0]["content"] if msgs[0]["role"] == "system" else ""

            comp_msgs, meta = compressor.compress_messages(msgs, system_text=system_text)

            orig_tokens = sum(measure_tokens(m.get("content", "")) for m in msgs)
            comp_tokens = sum(measure_tokens(m.get("content", "")) for m in comp_msgs)
            savings = round((1 - comp_tokens / max(1, orig_tokens)) * 100, 1)

            report = {
                "scenario": name,
                "original_tokens_est": orig_tokens,
                "compressed_tokens_est": comp_tokens,
                "savings_pct": savings,
                "messages_in": len(msgs),
                "messages_out": len(comp_msgs),
                "compressed": meta.get("compressed", False),
            }
            level_results.append(report)

            print(f"\n  {name}")
            print(f"    Original:  ~{orig_tokens} tokens")
            print(f"    Compressed: ~{comp_tokens} tokens")
            print(f"    Savings:    {savings}%")
            print(f"    Messages:   {len(msgs)} → {len(comp_msgs)}")
            print(f"    Compressed: {meta.get('compressed', False)}")

        total_orig = sum(r["original_tokens_est"] for r in level_results)
        total_comp = sum(r["compressed_tokens_est"] for r in level_results)
        avg_pct = round((1 - total_comp / max(1, total_orig)) * 100, 1)

        print(f"\n  {'=' * 50}")
        print(f"  TOTAL: ~{total_orig} tokens → ~{total_comp} tokens")
        print(f"  Savings: {avg_pct}% ({total_orig - total_comp} tokens saved)")
        print(f"  {'=' * 50}")

        all_results.append({
            "level": level.value,
            "total_original": total_orig,
            "total_compressed": total_comp,
            "total_savings_pct": avg_pct,
            "scenarios": level_results,
        })

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY — v2 Compression by Level")
    print(f"{'=' * 70}")
    print(f"{'Level':<15} {'Original':>10} {'Compressed':>12} {'Savings':>10}")
    print(f"{'─' * 50}")
    for r in all_results:
        print(f"{r['level']:<15} ~{r['total_original']:>8} ~{r['total_compressed']:>10} {r['total_savings_pct']:>9}%")

    # ── v1 vs v2 comparison ──
    print(f"\n{'=' * 70}")
    print("v1 vs v2 Comparison")
    print(f"{'=' * 70}")
    print(f"  v1 (MODERATE):  ~1367 → ~1192 = 12.8% savings")
    for r in all_results:
        print(f"  v2 ({r['level']:<10}):  ~{r['total_original']} → ~{r['total_compressed']} = {r['total_savings_pct']}% savings")

    # ── MiMo cost comparison ──
    print(f"\n{'=' * 70}")
    print("MiMo V2.5 Cost (Input $1.00/M, Cache Hit $0.20/M)")
    print(f"{'=' * 70}")

    MIMO_INPUT = 1.00 / 1_000_000
    MIMO_CACHE = 0.20 / 1_000_000
    MIMO_WRITE = 0.0

    total_raw = all_results[0]["total_original"]
    raw_cost = total_raw * MIMO_INPUT

    print(f"\n  Raw (no optimization): ${raw_cost:.6f} per batch")
    for r in all_results:
        comp = r["total_compressed"]
        cached = int(comp * 0.80)
        uncached = comp - cached
        cost = cached * MIMO_CACHE + uncached * MIMO_INPUT
        save = raw_cost - cost
        pct = round(save / raw_cost * 100, 1) if raw_cost > 0 else 0
        print(f"  v2+L0 ({r['level']:<10}): ${cost:.6f} (save {pct}%)")

    results_path = os.path.join(os.path.dirname(__file__), "l1_benchmark_v2_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    run_benchmark()
