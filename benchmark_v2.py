#!/usr/bin/env python3
"""Benchmark v2: Long-prefix test — triggers real MiMo cache (1024+ tokens)."""

import json
import time
import hashlib
import httpx

API_BASE = "https://token-plan-cn.xiaomimimo.com/v1"
API_KEY = "tp-c5t6lpqsivln86cec7lwwepjo3ngob6thblr34mxwddxy084"
MODEL = "mimo-v2.5"

# Long system prompt (~1500 tokens)
SYSTEM_PROMPT = """You are an expert AI coding assistant. You specialize in Python, JavaScript, and systems programming.

## Core Instructions
1. Always provide working code examples
2. Explain your reasoning before writing code
3. Use type hints in Python code
4. Follow PEP 8 style guidelines
5. Include error handling where appropriate
6. Prefer standard library over third-party packages when possible
7. When refactoring, maintain backward compatibility
8. Write docstrings for all public functions
9. Use meaningful variable and function names
10. Keep functions under 20 lines when possible

## Code Style
- Python: Use f-strings, pathlib, dataclasses where appropriate
- JavaScript: Use async/await, destructuring, optional chaining
- Always handle edge cases (None, empty strings, zero division)
- Use context managers for file/resource operations

## Communication Style
- Be concise but thorough
- Lead with the answer, then explain
- Use code blocks with language tags
- Point out potential pitfalls proactively
- Suggest related best practices when relevant

## Tools Available
- search_web: Search the internet for current information
- read_file: Read file contents from the project
- write_file: Create or modify files
- execute_code: Run Python code in a sandbox
- get_weather: Get current weather data for a city
- lookup_docs: Look up API documentation for libraries

When using tools, always explain what you're doing and why.
After receiving tool results, integrate them into your response naturally.
Never make up tool results — always use the actual output."""

# History
HISTORY = []
names = ['parser','cache','queue','tree','graph','sort','filter','adapter']
titles = ['Parser','Cache','Queue','Tree','Graph','Sort','Filter','Adapter']
for i in range(8):
    HISTORY.extend([
        {"role": "user", "content": f"Question {i+1}: Can you help me implement a {names[i]} in Python?"},
        {"role": "assistant", "content": f"Sure! Here is how to implement a {titles[i]}:\n\n```python\nclass {titles[i]}:\n    def __init__(self):\n        self.data = []\n    def process(self, item):\n        self.data.append(item)\n        return self.transform(item)\n    def transform(self, item):\n        return item\n```\n\nKey principles: encapsulation, single responsibility, extensibility."},
    ])

QUERIES = [
    "Add a method to serialize the data to JSON",
    "How do I add logging to this class?",
    "Can you write unit tests for the process method?",
    "What about adding async support?",
    "Show me how to add retry logic",
]

TOOLS = [
    {"type": "function", "function": {"name": "search_web", "description": "Search internet", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "execute_code", "description": "Run code", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "lookup_docs", "description": "Docs", "parameters": {"type": "object", "properties": {"library": {"type": "string"}}}}},
]
SORTED_TOOLS = sorted(TOOLS, key=lambda t: t.get("function", {}).get("name", ""))


def prefix_hash(messages):
    return hashlib.sha256(json.dumps(messages[:-1], sort_keys=True).encode()).hexdigest()[:16]


def call_api(messages, tools=None):
    payload = {"model": MODEL, "messages": messages, "max_tokens": 150}
    if tools:
        payload["tools"] = tools
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    with httpx.Client(timeout=30) as client:
        resp = client.post(f"{API_BASE}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


def run():
    print("=" * 60)
    print("  Benchmark v2 — Long Prefix (1024+ tokens)")
    print("=" * 60)

    # Phase 1: Raw (system at end)
    print("\n  Phase 1: RAW (system prompt at END)")
    raw_hashes, raw_tokens = [], []
    for i, q in enumerate(QUERIES):
        msgs = [{"role": "user", "content": q}, *HISTORY, {"role": "system", "content": SYSTEM_PROMPT}]
        h = prefix_hash(msgs)
        t0 = time.time()
        try:
            resp = call_api(msgs, SORTED_TOOLS)
            lat = time.time() - t0
            pt = resp.get("usage", {}).get("prompt_tokens", 0)
            raw_hashes.append(h)
            raw_tokens.append(pt)
            print(f"    Q{i+1}: hash={h} tokens={pt} {lat:.1f}s")
        except Exception as e:
            print(f"    Q{i+1}: ERROR {e}")
        time.sleep(1.5)

    # Phase 2: Optimized (system at front)
    print("\n  Phase 2: OPTIMIZED (system prompt FIRST)")
    opt_hashes, opt_tokens = [], []
    for i, q in enumerate(QUERIES):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *HISTORY, {"role": "user", "content": q}]
        h = prefix_hash(msgs)
        t0 = time.time()
        try:
            resp = call_api(msgs, SORTED_TOOLS)
            lat = time.time() - t0
            pt = resp.get("usage", {}).get("prompt_tokens", 0)
            opt_hashes.append(h)
            opt_tokens.append(pt)
            print(f"    Q{i+1}: hash={h} tokens={pt} {lat:.1f}s")
        except Exception as e:
            print(f"    Q{i+1}: ERROR {e}")
        time.sleep(1.5)

    # Results
    raw_u = len(set(raw_hashes))
    opt_u = len(set(opt_hashes))
    avg_raw = sum(raw_tokens) / max(len(raw_tokens), 1)
    avg_opt = sum(opt_tokens) / max(len(opt_tokens), 1)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n  RAW:    {raw_u}/{len(raw_hashes)} unique prefixes → cache {'0% ❌' if raw_u > 1 else 'possible'}")
    print(f"  OPT:    {opt_u}/{len(opt_hashes)} unique prefixes → cache {'up to ' + str(int((1-opt_u/max(len(opt_hashes),1))*100)) + '% ✅' if opt_u <= 1 else 'partial'}")
    print(f"  Tokens: {avg_raw:.0f} → {avg_opt:.0f} ({(1-avg_opt/avg_raw)*100:.1f}% less)")

    if raw_tokens:
        raw_cost = avg_raw / 1e6 * 1.00
        opt_cost = avg_opt / 1e6 * 0.20
        print(f"\n  MiMo Cost per request:")
        print(f"    Raw:      ${raw_cost:.8f} (full input rate)")
        print(f"    Optimized: ${opt_cost:.8f} (cache hit rate)")
        print(f"    Savings:   {(1-opt_cost/raw_cost)*100:.1f}%")
    print()


if __name__ == "__main__":
    run()
