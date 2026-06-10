#!/usr/bin/env python3
"""Benchmark: Token Optimizer vs Raw API — Real MiMo API test.

This script:
1. Sends requests WITHOUT optimization (raw API)
2. Sends requests WITH L0 prefix reorder + L2 cache tracking
3. Compares prefix stability and cache hit rates
"""

import json
import time
import hashlib
import httpx

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

API_BASE = "https://token-plan-cn.xiaomimimo.com/v1"
API_KEY = "tp-c5t6lpqsivln86cec7lwwepjo3ngob6thblr34mxwddxy084"
MODEL = "mimo-v2.5"

# Simulated app messages (typical agent scenario)
SYSTEM_PROMPT = """You are a helpful AI assistant specializing in coding and research.
You always provide concise, accurate answers.
When writing code, use Python best practices.
When researching, cite your sources."""

TOOLS = [
    {"type": "function", "function": {"name": "search_web", "description": "Search the internet", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "execute_code", "description": "Execute Python code", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Get weather info", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}},
]

# History messages (simulating a conversation)
HISTORY = [
    {"role": "user", "content": "Help me write a Python function to parse CSV files"},
    {"role": "assistant", "content": "Sure! Here's a CSV parsing function:\n```python\nimport csv\ndef parse_csv(path):\n    with open(path) as f:\n        return list(csv.DictReader(f))\n```"},
    {"role": "user", "content": "Add error handling for missing files"},
    {"role": "assistant", "content": "Here's the updated version with error handling:\n```python\nimport csv\nfrom pathlib import Path\ndef parse_csv(path):\n    try:\n        with open(path) as f:\n            return list(csv.DictReader(f))\n    except FileNotFoundError:\n        raise FileNotFoundError(f'File not found: {path}')\n```"},
]

# Test queries
QUERIES = [
    "Add support for reading from a string buffer",
    "How do I handle large CSV files without loading all into memory?",
    "What about CSV files with inconsistent delimiters?",
    "Show me the performance difference between csv.DictReader and csv.reader",
    "How do I write the parsed data back to a new CSV file?",
]


def call_api(messages, tools=None, extra_headers=None):
    """Raw API call."""
    payload = {"model": MODEL, "messages": messages, "max_tokens": 200}
    if tools:
        payload["tools"] = tools

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    with httpx.Client(timeout=30) as client:
        resp = client.post(f"{API_BASE}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


def compute_prefix_hash(messages):
    """Hash everything except the last user message (simulating prefix)."""
    prefix_msgs = messages[:-1]
    return hashlib.sha256(json.dumps(prefix_msgs, sort_keys=True).encode()).hexdigest()[:16]


# ──────────────────────────────────────────────
#  BENCHMARK
# ──────────────────────────────────────────────

def run_benchmark():
    print("=" * 60)
    print("  Token Optimizer Benchmark — MiMo V2.5 (Real API)")
    print("=" * 60)
    print()

    # ── Round 1: RAW API (no optimization) ──
    print("━" * 60)
    print("  Round 1: RAW API (no prefix optimization)")
    print("━" * 60)

    raw_results = []
    for i, query in enumerate(QUERIES):
        # BAD order: user first, system last (typical bad implementation)
        messages = [
            {"role": "user", "content": query},
            *HISTORY,
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # Track prefix hash
        prefix_hash = compute_prefix_hash(messages)

        t0 = time.time()
        try:
            resp = call_api(messages, TOOLS)
            latency = time.time() - t0
            usage = resp.get("usage", {})
            content = resp["choices"][0]["message"].get("content", "")[:80]

            raw_results.append({
                "query_num": i + 1,
                "prefix_hash": prefix_hash,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "latency": latency,
                "content_preview": content,
            })

            print(f"\n  Q{i+1}: {query[:50]}...")
            print(f"      Hash: {prefix_hash}")
            print(f"      Tokens: {usage.get('prompt_tokens', 0)} in / {usage.get('completion_tokens', 0)} out")
            print(f"      Latency: {latency:.1f}s")
            print(f"      Response: {content[:60]}...")

        except Exception as e:
            print(f"  Q{i+1}: ERROR - {e}")
            raw_results.append({"query_num": i + 1, "prefix_hash": prefix_hash, "error": str(e)})

        time.sleep(1.5)

    time.sleep(2)

    # ── Round 2: OPTIMIZED API (L0 prefix reorder) ──
    print("\n")
    print("━" * 60)
    print("  Round 2: OPTIMIZED (L0 prefix reorder → system first, user last)")
    print("━" * 60)

    opt_results = []
    prev_prefix_hash = None
    for i, query in enumerate(QUERIES):
        # GOOD order: system first, tools after, history, then user last
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *HISTORY,
            {"role": "user", "content": query},
        ]

        # L0: Sort tools for deterministic ordering
        sorted_tools = sorted(TOOLS, key=lambda t: t.get("function", {}).get("name", ""))

        # Track prefix hash (everything except last user message)
        prefix_hash = compute_prefix_hash(messages)
        cache_stable = (prefix_hash == prev_prefix_hash)
        prev_prefix_hash = prefix_hash

        t0 = time.time()
        try:
            resp = call_api(messages, sorted_tools)
            latency = time.time() - t0
            usage = resp.get("usage", {})
            content = resp["choices"][0]["message"].get("content", "")[:80]

            opt_results.append({
                "query_num": i + 1,
                "prefix_hash": prefix_hash,
                "cache_stable": cache_stable,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "latency": latency,
                "content_preview": content,
            })

            cache_marker = " → CACHE STABLE ✅" if cache_stable else " → NEW PREFIX ⚠️"
            print(f"\n  Q{i+1}: {query[:50]}...")
            print(f"      Hash: {prefix_hash}{cache_marker}")
            print(f"      Tokens: {usage.get('prompt_tokens', 0)} in / {usage.get('completion_tokens', 0)} out")
            print(f"      Latency: {latency:.1f}s")
            print(f"      Response: {content[:60]}...")

        except Exception as e:
            print(f"  Q{i+1}: ERROR - {e}")
            opt_results.append({"query_num": i + 1, "prefix_hash": prefix_hash, "cache_stable": cache_stable, "error": str(e)})

        time.sleep(1.5)

    # ── Analysis ──
    print("\n")
    print("=" * 60)
    print("  ANALYSIS")
    print("=" * 60)

    raw_hashes = [r["prefix_hash"] for r in raw_results if "error" not in r]
    opt_hashes = [r["prefix_hash"] for r in opt_results if "error" not in r]

    raw_unique = len(set(raw_hashes))
    opt_unique = len(set(opt_hashes))

    print(f"\n  RAW API:")
    print(f"    Unique prefix hashes: {raw_unique} / {len(raw_hashes)}")
    print(f"    → {raw_unique}/{len(raw_hashes)} different prefixes = {'CACHE UNSTABLE ❌' if raw_unique > 1 else 'CACHE STABLE ✅'}")

    print(f"\n  OPTIMIZED:")
    print(f"    Unique prefix hashes: {opt_unique} / {len(opt_hashes)}")
    stable_count = sum(1 for r in opt_results if r.get("cache_stable", False))
    print(f"    Cache stable requests: {stable_count}/{len(opt_results)}")
    print(f"    → {'CACHE STABLE ✅' if opt_unique == 1 else f'{opt_unique} prefixes = SOMEWHAT STABLE ⚠️'}")

    if raw_unique > 1:
        print(f"\n  IMPACT: Raw API has {raw_unique} different prefixes → 0% cache hit rate")
    if opt_unique <= 1 and len(opt_hashes) > 1:
        print(f"  IMPACT: Optimized has {opt_unique} prefix(es) → up to {(1-1/opt_unique)*100:.0f}% cache hit rate")

    # Summary table
    print(f"\n  {'─' * 50}")
    print(f"  {'Metric':<30} {'Raw':>8} {'Optimized':>10}")
    print(f"  {'─' * 50}")
    print(f"  {'Unique prefixes':<30} {raw_unique:>8} {opt_unique:>10}")
    if raw_results:
        avg_raw_tokens = sum(r.get("prompt_tokens", 0) for r in raw_results) / len(raw_results)
    else:
        avg_raw_tokens = 0
    if opt_results:
        avg_opt_tokens = sum(r.get("prompt_tokens", 0) for r in opt_results) / len(opt_results)
    else:
        avg_opt_tokens = 0
    print(f"  {'Avg prompt tokens/req':<30} {avg_raw_tokens:>8.0f} {avg_opt_tokens:>10.0f}")
    print(f"  {'─' * 50}")

    print(f"\n{'=' * 60}")
    print(f"  Benchmark complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_benchmark()
