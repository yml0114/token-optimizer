#!/usr/bin/env python3
"""Benchmark: Headroom integration test v3."""
import sys, os, json, time, statistics as st
os.chdir('/Users/liangliang/.qwenpaw/workspaces/default/token-optimizer')
sys.path.insert(0, 'src')

from token_optimizer.core.compression_store import CompressionStore
from token_optimizer.core.prompt_reorderer import strip_dynamic_fields, compute_prefix_hash, DynamicContentDetector
from token_optimizer.core.smart_compressor import SmartCompressor, StatisticalAnalyzer, estimate_tokens_from_messages
from token_optimizer.core.signal_noise import InputCompressor, CompressionLevel

# ── Scenarios ───────────────────────────────────────────────────────────────

def scenario_short():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]

def scenario_medium():
    return [
        {"role": "system", "content": "You are an expert Python developer. Help users write clean, efficient code."},
        {"role": "user", "content": "Implement quicksort in Python with type hints and None handling."},
        {"role": "assistant", "content": "Here's quicksort:\n```python\ndef quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    return quicksort([x for x in arr if x < pivot]) + [x for x in arr if x == pivot] + quicksort([x for x in arr if x > pivot])\n```\nO(n log n) average."},
        {"role": "user", "content": "Handle edge cases: empty list and all-None arrays."},
    ]

def scenario_long():
    return [
        {"role": "system", "content": "You are a senior software architect. Provide detailed technical guidance on system design, focusing on scalability, maintainability, and performance optimization."},
        {"role": "user", "content": "Design microservice architecture for e-commerce: 10K concurrent users, real-time orders, inventory consistency across warehouses."},
        {"role": "assistant", "content": "Core Services:\n1. User Service: JWT + Redis sessions\n2. Product Catalog: Elasticsearch\n3. Inventory: Event sourcing + Kafka\n4. Order: Saga pattern\n5. Payment: Idempotency keys\n6. Notification: Async queues\n\nInfra: Kong gateway, Istio mesh, Kafka, Redis cluster, PostgreSQL + MongoDB, Kubernetes."},
        {"role": "user", "content": "How to prevent overselling when two users buy the same last item?"},
        {"role": "assistant", "content": "Multi-layer:\n1. Optimistic Locking: UPDATE SET qty=qty-1 WHERE qty>0 AND version=?\n2. Redis Lock: async with redis.lock(f'inv:{pid}:{wid}'):\n3. Event Sourcing: Kafka events\n4. Reservation TTL: 15min"},
        {"role": "user", "content": "Explain saga pattern for order flow."},
    ]

def scenario_json_dict():
    """JSON dict — StatisticalAnalyzer analyzes fields but doesn't compress."""
    return [
        {"role": "system", "content": "You are a data analyst assistant."},
        {"role": "user", "content": "Analyze this API response for anomalies."},
        {"role": "tool", "content": json.dumps({"status":"success","request_id":"req_7f3a2b1c-9d4e","timestamp":"2026-06-11T03:45:00Z","data":{"metrics":[{"name":"cpu","value":45.2},{"name":"mem","value":78.9},{"name":"disk","value":234.5}]}})},
    ]

def scenario_json_list():
    """JSON list-of-dicts — StatisticalAnalyzer can CSV+Schema compress."""
    records = [{"id": i, "name": f"user_{i}", "score": round(85 + i * 0.3, 1), "active": i % 3 != 0} for i in range(50)]
    return [
        {"role": "system", "content": "You are a data analyst."},
        {"role": "user", "content": "Summarize this user data."},
        {"role": "tool", "content": json.dumps(records)},
    ]

def scenario_code():
    return [
        {"role": "system", "content": "You are a Python expert."},
        {"role": "user", "content": "Review:\n```python\nimport os, json\nclass DP:\n    def __init__(self, p):\n        with open(p) as f: self.c = json.load(f)\n    def proc(self, d):\n        r=[]\n        for fn in os.listdir(d):\n            if fn.endswith('.json'):\n                with open(os.path.join(d,fn)) as f: r.append(json.load(f))\n        return r\n```"},
    ]

# ── Benchmark Engine ────────────────────────────────────────────────────────

def bench(msgs, n=5):
    orig = estimate_tokens_from_messages(msgs)
    rule = InputCompressor(level=CompressionLevel.AGGRESSIVE)
    analyzer = StatisticalAnalyzer()

    br, bl, er, el = [], [], [], []

    for _ in range(n):
        # Baseline: rule-only
        t0 = time.perf_counter()
        c, _ = rule.compress_messages(msgs)
        dt = (time.perf_counter() - t0) * 1000
        ct = estimate_tokens_from_messages(c)
        br.append(ct / orig if orig > 0 else 1.0)
        bl.append(dt)

        # Enhanced: rule + StatisticalAnalyzer
        t0 = time.perf_counter()
        enhanced_msgs = []
        for m in msgs:
            content = m.get("content", "")
            new_content = content
            if isinstance(content, str) and content.strip().startswith(("{", "[")):
                try:
                    optimized, _ = analyzer.analyze_and_compress_text(content)
                    new_content = optimized
                except Exception:
                    pass
            enhanced_msgs.append({**m, "content": new_content})
        c2, _ = rule.compress_messages(enhanced_msgs)
        dt = (time.perf_counter() - t0) * 1000
        ct2 = estimate_tokens_from_messages(c2)
        er.append(ct2 / orig if orig > 0 else 1.0)
        el.append(dt)

    def s(a):
        return {"mean": round(st.mean(a), 4), "std": round(st.stdev(a) if len(a) > 1 else 0, 4)}

    return {"orig": orig, "baseline": {"ratio": s(br), "lat_ms": s(bl)}, "enhanced": {"ratio": s(er), "lat_ms": s(el)}}

# ── Main ────────────────────────────────────────────────────────────────────

print("=" * 60)
print("HEADROOM INTEGRATION BENCHMARK v3")
print("=" * 60)

results = {}
for name, fn in [("short", scenario_short), ("medium", scenario_medium),
                  ("long", scenario_long), ("json_dict", scenario_json_dict),
                  ("json_list", scenario_json_list), ("code", scenario_code)]:
    print(f"▸ {name}...", end=" ", flush=True)
    results[name] = bench(fn(), 5)
    b = results[name]["baseline"]["ratio"]["mean"]
    e = results[name]["enhanced"]["ratio"]["mean"]
    d = results[name]["baseline"]["lat_ms"]["mean"]
    print(f"base={b:.3f} enh={e:.3f} Δ={round((b-e)*100,1):+.1f}% lat_base={d:.1f}ms")

# Dynamic prefix stability
print("▸ dynamic_prefix...", end=" ", flush=True)
base = "You are a production monitoring assistant. Current date: {date}. Session: {session}. Trace: {trace}. System version: v2.4.1 stable."
ma = [
    {"role": "system", "content": base.format(date="2026-06-11", session="session_abc123def456", trace="req_7f3a2b1c-9d4e")},
    {"role": "user", "content": "Check server health."},
]
mb = [
    {"role": "system", "content": base.format(date="2026-06-12", session="session_xyz789ghi012", trace="req_1a2b3c4d-5e6f")},
    {"role": "user", "content": "What's the error rate?"},
]
ca, cb = strip_dynamic_fields(ma), strip_dynamic_fields(mb)
ha, hb = compute_prefix_hash(ca), compute_prefix_hash(cb)
print(f"stable={ha == hb}")
results["prefix"] = {"stable": ha == hb, "a": ha[:16], "b": hb[:16]}

# CCR store/retrieve
print("▸ CCR...", end=" ", flush=True)
store = CompressionStore()
ok = 0
for t in ["Long text " * 50, '{"k":"v","a":[1,2,3]}' * 10, "Short"]:
    h, ann = store.store(t, "comp")
    r = store.retrieve(h)
    if r == t:
        ok += 1
print(f"{ok}/3 retrieved, stats={store.stats}")
results["ccr"] = {"ok": ok, "total": 3, "stats": store.stats, "hit_rate": store.get_hit_rate()}

# SmartCompressor integrated test (CCR-aware compression)
print("▸ SmartCompressor+CCR...", end=" ", flush=True)
sc = SmartCompressor()
test_msgs = scenario_long()
comp_msgs, comp_meta = sc.compress(test_msgs)
orig_t = estimate_tokens_from_messages(test_msgs)
comp_t = estimate_tokens_from_messages(comp_msgs)
print(f"tokens: {orig_t} → {comp_t} (Δ={round((1-comp_t/orig_t)*100,1)}%)")

print("\n" + "=" * 60)
print("FULL RESULTS (JSON)")
print("=" * 60)
print(json.dumps(results, indent=2, ensure_ascii=False))
print("\nDONE ✅")
