"""Benchmark L1 v4: Word-level filtering + history compression v2.

Key v4 additions:
  - Long conversations (10+ turns) to test history compression
  - Cross-turn instruction dedup scenarios
  - Word-level filler scenarios (那么/顺便/能加/其实)
  - Real-world noisy Chinese conversation patterns

Run: python benchmark_l1_v4.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from token_optimizer.core.signal_noise import (
    InputCompressor,
    CompressionLevel,
)


def measure_tokens(text: str) -> int:
    return max(1, len(text) // 3)


SYSTEM_PROMPT = """You are a senior Python developer. Write clean, well-documented code.
Always use type hints. Prefer dataclasses over dicts for structured data.
Error handling: use specific exceptions, not bare except.
Testing: write tests for all public functions.
Code style: PEP 8 compliant, 4-space indentation, max line length 88.
When asked to explain, be concise -- 3 sentences max."""


CONVERSATIONS = [
    # ── Original v3 scenarios (keep for comparison) ──
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
            {"role": "tool", "content": "HTTP/1.1 200 OK\nContent-Type: application/json\nX-RateLimit-Remaining: 59\nX-RateLimit-Reset: 1718064000\nrequest_id: abc123def456\ntrace_id: xyz789\nlatency: 45ms\n{\"sha\":\"a1b2c3d\",\"message\":\"feat: add L1 compression\",\"date\":\"2026-06-10\"}"},
            {"role": "user", "content": "看起来不错，请问这个commit具体做了什么改动？如果可以的话详细说一下。"},
        ],
    },
    {
        "name": "Error debugging with stack trace",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "运行出错了，帮我看看。不好意思，之前忘了检查类型。"},
            {"role": "assistant", "content": "File \"main.py\", line 42\n    result = process_data(raw_input)\nFile \"utils.py\", line 15\n    return {k: v.strip() for k, v in data.items()}\nAttributeError: 'NoneType' object has no attribute 'strip'"},
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
    {
        "name": "[v3] Heavy filler user with transition words",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "好的谢谢，请帮我写一个完整的缓存系统，如果可以的话用Python实现。"},
            {"role": "assistant", "content": "class Cache:\n    def __init__(self):\n        self._data = {}\n    def get(self, key):\n        return self._data.get(key)\n    def put(self, key, val):\n        self._data[key] = val"},
            {"role": "user", "content": "很好，那么接下来能加一个LRU淘汰策略吗？麻烦你了，如果可以的话顺便加一个命中率统计功能。"},
            {"role": "assistant", "content": "from collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, max_size=100):\n        self._cache = OrderedDict()\n        self._max_size = max_size\n        self._hits = 0\n        self._misses = 0\n    \n    def get(self, key):\n        if key not in self._cache:\n            self._misses += 1\n            return None\n        self._hits += 1\n        self._cache.move_to_end(key)\n        return self._cache[key]\n    \n    def put(self, key, val):\n        if key in self._cache:\n            self._cache.move_to_end(key)\n        elif len(self._cache) >= self._max_size:\n            self._cache.popitem(last=False)\n        self._cache[key] = val\n    \n    @property\n    def hit_rate(self):\n        total = self._hits + self._misses\n        return self._hits / total if total > 0 else 0.0"},
            {"role": "user", "content": "太好了！顺便说一下，如果能加一个过期时间(TTL)功能就更好了。好的谢谢！"},
        ],
    },
    {
        "name": "[v3] Politeness-heavy with repeated noise",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "麻烦你帮我创建一个完整的、功能齐全的用户认证模块，如果可以的话用JWT实现，谢谢了！"},
            {"role": "assistant", "content": "class AuthModule:\n    def __init__(self, secret: str):\n        self._secret = secret\n    def create_token(self, user_id: str) -> str:\n        return jwt.encode({\"uid\": user_id}, self._secret, algorithm=\"HS256\")\n    def verify_token(self, token: str) -> dict | None:\n        try:\n            return jwt.decode(token, self._secret, algorithms=[\"HS256\"])\n        except:\n            return None"},
            {"role": "user", "content": "很好！接下来请帮我加一个角色权限检查功能，如果可以的话用装饰器实现，麻烦你了，辛苦了！"},
        ],
    },

    # ── v4: NEW long conversation scenarios (10+ turns) ──
    {
        "name": "[v4] Long coding session (10 turns)",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "请帮我写一个排序算法，如果可以的话用Python。"},
            {"role": "assistant", "content": "def sort_array(arr: list[int]) -> list[int]:\n    return sorted(arr)"},
            {"role": "user", "content": "好的谢谢！那么能不能加一个自定义比较函数的支持？"},
            {"role": "assistant", "content": "from typing import Callable\n\ndef sort_array(arr: list[int], key: Callable[[int], int] = lambda x: x) -> list[int]:\n    return sorted(arr, key=key)"},
            {"role": "user", "content": "确实不错。接下来能加一个原地排序的版本吗？"},
            {"role": "assistant", "content": "def sort_inplace(arr: list[int]) -> None:\n    arr.sort()"},
            {"role": "user", "content": "很好！顺便说一下，如果能加一个稳定性检查功能就更好了。"},
            {"role": "assistant", "content": "def is_stable_sort(sort_fn, test_arr: list[int] = None) -> bool:\n    if test_arr is None:\n        test_arr = [(3, 'a'), (1, 'b'), (3, 'c'), (1, 'd')]\n    pairs = list(enumerate(test_arr))\n    sorted_pairs = sorted(pairs, key=lambda x: x[1])\n    indices = [p[0] for p in sorted_pairs]\n    return indices == sorted(indices)"},
            {"role": "user", "content": "太好了！麻烦你帮我写一个单元测试来验证所有排序功能。"},
            {"role": "assistant", "content": "import pytest\n\ndef test_sort_basic():\n    assert sort_array([3, 1, 2]) == [1, 2, 3]\n\ndef test_sort_with_key():\n    assert sort_array([3, 1, 2], key=lambda x: -x) == [3, 2, 1]\n\ndef test_sort_inplace():\n    arr = [3, 1, 2]\n    sort_inplace(arr)\n    assert arr == [1, 2, 3]\n\ndef test_stability():\n    assert is_stable_sort(sort_array)"},
            {"role": "user", "content": "好的谢谢！请帮我写一个性能基准测试，麻烦你了。"},
        ],
    },
    {
        "name": "[v4] Long debugging session with repeated instructions",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "帮我看看这个bug"},
            {"role": "tool", "content": "HTTP/1.1 500 Internal Server Error\nContent-Type: application/json\nX-Trace: debug-trace-001\n{\"error\": \"NullPointerException\", \"file\": \"main.java\", \"line\": 42}"},
            {"role": "assistant", "content": "NullPointerException at line 42 in main.java. Check if the variable is null before accessing it."},
            {"role": "user", "content": "请帮我修复这个bug，加一个null检查"},
            {"role": "assistant", "content": "// Fixed: added null check\nif (obj != null) {\n    obj.doSomething();\n}"},
            {"role": "user", "content": "那么还有另一个bug，请帮我也修一下"},
            {"role": "tool", "content": "HTTP/1.1 500 Internal Server Error\nContent-Type: application/json\nX-Trace: debug-trace-002\n{\"error\": \"IndexOutOfBounds\", \"file\": \"utils.java\", \"line\": 15}"},
            {"role": "assistant", "content": "IndexOutOfBounds at line 15 in utils.java. Array size check needed before access."},
            {"role": "user", "content": "能不能加一个单元测试来验证这两个修复？"},
            {"role": "assistant", "content": "@Test\npublic void testNullCheck() {\n    assertNull(obj); // Should not throw\n}\n\n@Test\npublic void testArrayBounds() {\n    int[] arr = new int[5];\n    assertEquals(0, arr[4]); // Should not throw\n}"},
            {"role": "user", "content": "很好！接下来请帮我重构一下整个错误处理模块，麻烦你了。"},
            {"role": "assistant", "content": "class ErrorHandler {\n    public void handle(Exception e) {\n        if (e instanceof NullPointerException) {\n            log.error(\"Null pointer\", e);\n        } else if (e instanceof IndexOutOfBoundsException) {\n            log.error(\"Index out of bounds\", e);\n        }\n    }\n}"},
            {"role": "user", "content": "请帮我写一个性能测试"},
        ],
    },
    {
        "name": "[v4] v4 word-level filler scenario",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "那么我想写一个数据库连接池，其实就是一个连接管理器。能不能加一个连接超时配置？就是说连接超过30秒没用就自动关闭那种。"},
            {"role": "assistant", "content": "import time\nfrom threading import Lock\n\nclass ConnectionPool:\n    def __init__(self, max_size=10, timeout=30):\n        self._pool = []\n        self._max_size = max_size\n        self._timeout = timeout\n        self._lock = Lock()\n    \n    def get(self):\n        with self._lock:\n            if self._pool:\n                return self._pool.pop()\n        return self._create_connection()\n    \n    def _create_connection(self):\n        return {'id': id(self), 'created': time.time()}\n    \n    def put(self, conn):\n        with self._lock:\n            if len(self._pool) < self._max_size:\n                self._pool.append(conn)}"},
            {"role": "user", "content": "顺便说一下，能不能加一个健康检查功能？反正就是定期检查连接是否可用。"},
        ],
    },
]


def run_benchmark():
    print("=" * 70)
    print("L1 v4 Benchmark -- v4 Compression (word-level + history compression)")
    print("=" * 70)

    levels = [CompressionLevel.SAFE, CompressionLevel.MODERATE, CompressionLevel.AGGRESSIVE]
    all_results = []

    for level in levels:
        print(f"\n{'--' * 35}")
        print(f"Compression Level: {level.value.upper()}")
        print(f"{'--' * 35}")

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

            # History compression stats
            hist = meta.get("history_compression", {})
            hist_savings = hist.get("savings_pct", 0) if hist.get("compressed") else 0
            repeated = hist.get("repeated_instructions_removed", 0)

            report = {
                "scenario": name,
                "original_tokens_est": orig_tokens,
                "compressed_tokens_est": comp_tokens,
                "savings_pct": savings,
                "messages_in": len(msgs),
                "messages_out": len(comp_msgs),
                "compressed": meta.get("compressed", False),
                "history_savings_pct": hist_savings,
                "repeated_instructions_removed": repeated,
            }
            level_results.append(report)

            tag = ""
            if name.startswith("[v4]"):
                tag = " [v4]"
            elif name.startswith("[v3]"):
                tag = " [v3]"

            print(f"\n  {name}{tag}")
            print(f"    Original:  ~{orig_tokens} tokens")
            print(f"    Compressed: ~{comp_tokens} tokens")
            print(f"    Savings:    {savings}%", end="")
            if hist_savings > 0:
                print(f" (history: {hist_savings}%)", end="")
            if repeated > 0:
                print(f" [{repeated} deduped]", end="")
            print()

        total_orig = sum(r["original_tokens_est"] for r in level_results)
        total_comp = sum(r["compressed_tokens_est"] for r in level_results)
        avg_pct = round((1 - total_comp / max(1, total_orig)) * 100, 1)

        print(f"\n  {'=' * 50}")
        print(f"  TOTAL: ~{total_orig} tokens -> ~{total_comp} tokens")
        print(f"  Savings: {avg_pct}% ({total_orig - total_comp} tokens saved)")
        print(f"  {'=' * 50}")

        all_results.append({
            "level": level.value,
            "total_original": total_orig,
            "total_compressed": total_comp,
            "total_savings_pct": avg_pct,
            "scenarios": level_results,
        })

    # Summary
    print(f"\n{'=' * 70}")
    print("VERSION COMPARISON")
    print(f"{'=' * 70}")
    print(f"  v1 (MODERATE, 6 scenarios):     12.8% savings")
    print(f"  v2 (MODERATE, 6 scenarios):     23.5% savings")
    print(f"  v3 (AGGRESSIVE, 8 scenarios):   24.1% savings")
    for r in all_results:
        v4_scenarios = [s for s in r["scenarios"] if s["scenario"].startswith("[v4]")]
        v3_scenarios = [s for s in r["scenarios"] if s["scenario"].startswith("[v3]")]
        v2_scenarios = [s for s in r["scenarios"] if not s["scenario"].startswith("[v")]
        print(f"  v4 ({r['level']:<10}, 11 scenarios):  {r['total_savings_pct']}% savings")

    # MiMo cost analysis
    print(f"\n{'=' * 70}")
    print("MiMo V2.5 Cost (Input $1.00/M, Cache Hit $0.20/M)")
    print(f"{'=' * 70}")

    MIMO_INPUT = 1.00 / 1_000_000
    MIMO_CACHE = 0.20 / 1_000_000

    total_raw = all_results[2]["total_original"]  # AGGRESSIVE level
    raw_cost = total_raw * MIMO_INPUT

    print(f"\n  Raw (no optimization): ${raw_cost:.6f} per batch")
    for r in all_results:
        comp = r["total_compressed"]
        cached = int(comp * 0.80)
        uncached = comp - cached
        cost = cached * MIMO_CACHE + uncached * MIMO_INPUT
        save = raw_cost - cost
        pct = round(save / raw_cost * 100, 1) if raw_cost > 0 else 0
        print(f"  v4+L0 ({r['level']:<10}): ${cost:.6f} (save {pct}%)")

    # Per-scenario detail for AGGRESSIVE
    print(f"\n{'=' * 70}")
    print("AGGRESSIVE MODE - Per-Scenario Detail")
    print(f"{'=' * 70}")
    agg = all_results[2]
    for s in agg["scenarios"]:
        hist = s.get("history_savings_pct", 0)
        dedup = s.get("repeated_instructions_removed", 0)
        extra = ""
        if hist > 0:
            extra += f" hist={hist}%"
        if dedup > 0:
            extra += f" dedup={dedup}"
        print(f"  {s['scenario']:<50} {s['savings_pct']:>6}%{extra}")

    results_path = os.path.join(os.path.dirname(__file__), "l1_benchmark_v4_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    run_benchmark()
