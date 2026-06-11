#!/usr/bin/env python3
"""Benchmark script for Headroom integration into Token Optimizer.

Tests 6 scenarios comparing baseline (pre-integration) vs enhanced
(post-integration with CCR, CacheAligner, StatisticalAnalyzer).

Each scenario runs 5 iterations and collects mean ± std for:
- Token compression ratio
- Latency (ms)
- CCR retrieval success rate
- CacheAligner prefix stability
- Statistical analyzer accuracy
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

# Add project src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from token_optimizer.core.compression_store import CompressionStore
from token_optimizer.core.prompt_reorderer import (
    DynamicContentDetector,
    reorder_messages,
    strip_dynamic_fields,
    compute_prefix_hash,
)
from token_optimizer.core.smart_compressor import (
    SmartCompressor,
    StatisticalAnalyzer,
    estimate_tokens_from_messages,
)
from token_optimizer.core.signal_noise import InputCompressor, CompressionLevel


# ══════════════════════════════════════════════════════════════════════════════
# Test Scenarios
# ══════════════════════════════════════════════════════════════════════════════

def scenario_short_conversation() -> list[dict]:
    """Scenario 1: Short conversation (<100 tokens)."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]


def scenario_medium_conversation() -> list[dict]:
    """Scenario 2: Medium conversation (100-500 tokens)."""
    return [
        {"role": "system", "content": "You are an expert Python developer. Help users write clean, efficient code."},
        {"role": "user", "content": "I need help with a sorting algorithm. Can you implement quicksort in Python with good error handling?"},
        {"role": "assistant", "content": "Here's a quicksort implementation:\n```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n```\nThis uses the middle element as pivot. The time complexity is O(n log n) average case."},
        {"role": "user", "content": "Can you also add type hints and handle the case where the array contains None values?"},
    ]


def scenario_long_conversation() -> list[dict]:
    """Scenario 3: Long conversation (500+ tokens)."""
    return [
        {"role": "system", "content": "You are a senior software architect. Provide detailed technical guidance on system design, focusing on scalability, maintainability, and performance optimization."},
        {"role": "user", "content": "I'm designing a microservice architecture for an e-commerce platform. The system needs to handle 10,000 concurrent users, process orders in real-time, and maintain inventory consistency across multiple warehouses. Can you help me design the system?"},
        {"role": "assistant", "content": "Here's a comprehensive microservice architecture for your e-commerce platform:\n\n## Core Services\n1. **User Service**: Handles authentication, profiles, and session management. Use JWT tokens with Redis-backed session store.\n2. **Product Catalog Service**: Manages product information, categories, and search. Use Elasticsearch for full-text search.\n3. **Inventory Service**: Real-time inventory tracking across warehouses. Use event sourcing with Kafka for consistency.\n4. **Order Service**: Processes orders, manages order lifecycle. Use saga pattern for distributed transactions.\n5. **Payment Service**: Integrates with payment gateways. Use idempotency keys for retry safety.\n6. **Notification Service**: Sends emails, SMS, push notifications. Use message queues for async delivery.\n\n## Infrastructure\n- API Gateway: Kong or AWS API Gateway for routing, rate limiting, and auth\n- Service Mesh: Istio for service-to-service communication\n- Message Queue: Apache Kafka for event streaming\n- Cache: Redis cluster for session and hot data\n- Database: PostgreSQL for transactional data, MongoDB for product catalog\n- Container Orchestration: Kubernetes with auto-scaling\n\n## Consistency Model\nFor inventory consistency across warehouses, implement the Outbox Pattern with Kafka. Each warehouse publishes inventory change events to a dedicated topic. The Inventory Service consumes these events and maintains a materialized view."},
        {"role": "user", "content": "What about handling the case where two users try to purchase the same last item simultaneously? How do we prevent overselling?"},
        {"role": "assistant", "content": "To prevent overselling, implement a multi-layer approach:\n\n### Layer 1: Optimistic Locking\n```sql\nUPDATE inventory \nSET quantity = quantity - 1, version = version + 1\nWHERE product_id = ? AND warehouse_id = ? AND quantity > 0 AND version = ?\n```\nIf the UPDATE affects 0 rows, the item is already sold.\n\n### Layer 2: Distributed Lock with Redis\n```python\nasync def reserve_item(product_id: str, warehouse_id: str, user_id: str):\n    lock_key = f\"inventory:{product_id}:{warehouse_id}\"\n    async with redis.lock(lock_key, timeout=10):\n        inventory = await get_inventory(product_id, warehouse_id)\n        if inventory.quantity <= 0:\n            raise OutOfStockError()\n        await decrement_inventory(product_id, warehouse_id)\n        await create_reservation(product_id, warehouse_id, user_id)\n```\n\n### Layer 3: Event Sourcing\nAll inventory changes are events in Kafka. The canonical state is derived by replaying events. This provides audit trail and enables CQRS for read optimization.\n\n### Layer 4: Reservation TTL\nReservations expire after 15 minutes if payment is not completed, returning items to available pool."},
        {"role": "user", "content": "Can you explain how to implement the saga pattern for the order flow across payment, inventory, and shipping services?"},
    ]


def scenario_json_tool_output() -> list[dict]:
    """Scenario 4: Conversation with JSON tool output."""
    return [
        {"role": "system", "content": "You are a data analyst assistant. Help users analyze API response data."},
        {"role": "user", "content": "Analyze this API response and tell me if there are any anomalies."},
        {"role": "tool", "content": json.dumps({
            "status": "success",
            "request_id": "req_7f3a2b1c-9d4e-4f5a-8b6c-1e2d3f4a5b6c",
            "timestamp": "2026-06-11T03:45:00Z",
            "data": {
                "metrics": [
                    {"name": "cpu_usage", "value": 45.2, "unit": "%", "status": "normal"},
                    {"name": "memory_usage", "value": 78.9, "unit": "%", "status": "warning"},
                    {"name": "disk_io", "value": 234.5, "unit": "MB/s", "status": "normal"},
                    {"name": "network_latency", "value": 12.3, "unit": "ms", "status": "normal"},
                    {"name": "error_rate", "value": 0.02, "unit": "%", "status": "normal"},
                    {"name": "request_count", "value": 15234, "unit": "count", "status": "normal"},
                    {"name": "response_time_p99", "value": 450, "unit": "ms", "status": "warning"},
                    {"name": "active_connections", "value": 892, "unit": "count", "status": "normal"},
                ],
                "alerts": [
                    {"level": "warning", "message": "Memory usage above 75%", "timestamp": "2026-06-11T03:44:30Z"},
                    {"level": "warning", "message": "P99 response time above 400ms", "timestamp": "2026-06-11T03:44:45Z"},
                ],
            },
            "metadata": {
                "region": "us-east-1",
                "instance_id": "i-0abc123def456789",
                "collection_interval": "60s",
                "version": "2.4.1",
            },
        }, ensure_ascii=False, indent=2)},
    ]


def scenario_code_snippet() -> list[dict]:
    """Scenario 5: Conversation with code snippets."""
    return [
        {"role": "system", "content": "You are a Python expert. Help with code review and optimization."},
        {"role": "user", "content": "Review this code for performance issues:\n\n```python\nimport os\nimport json\nfrom typing import List, Dict, Optional\nfrom datetime import datetime, timedelta\n\nclass DataProcessor:\n    def __init__(self, config_path: str):\n        with open(config_path, 'r') as f:\n            self.config = json.load(f)\n        self.cache = {}\n        self.processed_count = 0\n    \n    def process_files(self, directory: str) -> List[Dict]:\n        results = []\n        for filename in os.listdir(directory):\n            filepath = os.path.join(directory, filename)\n            if not filename.endswith('.json'):\n                continue\n            with open(filepath, 'r') as f:\n                data = json.load(f)\n            processed = self._transform(data)\n            results.append(processed)\n            self.processed_count += 1\n        return results\n    \n    def _transform(self, data: Dict) -> Dict:\n        cache_key = str(data.get('id', ''))\n        if cache_key in self.cache:\n            return self.cache[cache_key]\n        \n        result = {\n            'id': data.get('id'),\n            'name': data.get('name', '').upper(),\n            'value': float(data.get('value', 0)) * 1.1,\n            'processed_at': datetime.now().isoformat(),\n            'expires_at': (datetime.now() + timedelta(days=30)).isoformat(),\n        }\n        \n        # Validate\n        if result['value'] < 0:\n            raise ValueError(f\"Negative value: {result['value']}\")\n        if not result['name']:\n            raise ValueError(\"Empty name\")\n        \n        self.cache[cache_key] = result\n        return result\n    \n    def get_stats(self) -> Dict:\n        return {\n            'processed_count': self.processed_count,\n            'cache_size': len(self.cache),\n            'timestamp': datetime.now().isoformat(),\n        }\n```\n\nIdentify all performance bottlenecks and suggest improvements."},
    ]


def scenario_dynamic_system_prompt() -> tuple[list[dict], list[dict]]:
    """Scenario 6: System prompt with dynamic content (dates, UUIDs, session IDs).

    Returns TWO message sets simulating consecutive requests with changing
    dynamic fields — used to test prefix stability.
    """
    base_system = (
        "You are a production monitoring assistant. "
        "Current date: {date}. "
        "Session ID: {session}. "
        "Request trace: {trace}. "
        "System version: v2.4.1 stable. "
        "You help engineers diagnose issues in real-time."
    )

    system_a = base_system.format(
        date="2026-06-11",
        session="session_abc123def456",
        trace="req_7f3a2b1c-9d4e-4f5a-8b6c-1e2d3f4a5b6c",
    )
    system_b = base_system.format(
        date="2026-06-12",
        session="session_xyz789ghi012",
        trace="req_1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d",
    )

    messages_a = [
        {"role": "system", "content": system_a},
        {"role": "user", "content": "Check the health status of production servers."},
    ]
    messages_b = [
        {"role": "system", "content": system_b},
        {"role": "user", "content": "What's the current error rate?"},
    ]
    return messages_a, messages_b


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark Functions
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_compression(messages: list[dict], iterations: int = 5) -> dict:
    """Benchmark rule-based compression (baseline) vs enhanced compression.

    Returns:
        Dict with baseline and enhanced compression metrics.
    """
    original_tokens = estimate_tokens_from_messages(messages)

    # Baseline: rule-only compression (v4)
    baseline_ratios = []
    baseline_latencies = []
    rule_compressor = InputCompressor(level=CompressionLevel.AGGRESSIVE)

    for _ in range(iterations):
        start = time.perf_counter()
        compressed, meta = rule_compressor.compress_messages(messages)
        elapsed = (time.perf_counter() - start) * 1000
        compressed_tokens = estimate_tokens_from_messages(compressed)
        ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        baseline_ratios.append(ratio)
        baseline_latencies.append(elapsed)

    # Enhanced: rule + StatisticalAnalyzer
    enhanced_ratios = []
    enhanced_latencies = []
    analyzer = StatisticalAnalyzer()

    for _ in range(iterations):
        start = time.perf_counter()
        compressed, meta = rule_compressor.compress_messages(messages)

        # Apply statistical analysis to any JSON content in messages
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip().startswith(("{", "[")):
                _, analyses = analyzer.analyze_and_compress_text(content)

        elapsed = (time.perf_counter() - start) * 1000
        compressed_tokens = estimate_tokens_from_messages(compressed)
        ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        enhanced_ratios.append(ratio)
        enhanced_latencies.append(elapsed)

    return {
        "original_tokens": original_tokens,
        "baseline": {
            "compression_ratio": {
                "mean": round(statistics.mean(baseline_ratios), 4),
                "std": round(statistics.stdev(baseline_ratios), 4) if len(baseline_ratios) > 1 else 0.0,
            },
            "latency_ms": {
                "mean": round(statistics.mean(baseline_latencies), 2),
                "std": round(statistics.stdev(baseline_latencies), 2) if len(baseline_latencies) > 1 else 0.0,
            },
        },
        "enhanced": {
            "compression_ratio": {
                "mean": round(statistics.mean(enhanced_ratios), 4),
                "std": round(statistics.stdev(enhanced_ratios), 4) if len(enhanced_ratios) > 1 else 0.0,
            },
            "latency_ms": {
                "mean": round(statistics.mean(enhanced_latencies), 2),
                "std": round(statistics.stdev(enhanced_latencies), 2) if len(enhanced_latencies) > 1 else 0.0,
            },
        },
    }


def benchmark_ccr(messages: list[dict], iterations: int = 5) -> dict:
    """Benchmark CCR (Compression with Content Recall) store.

    Tests:
    - Store success rate
    - Retrieve success rate
    - Store + retrieve latency
    - Annotated text correctness
    """
    store = CompressionStore(max_entries=50, default_ttl=300.0)

    store_latencies = []
    retrieve_latencies = []
    store_success_count = 0
    retrieve_success_count = 0
    total_texts = 0

    for _ in range(iterations):
        stored_hashes = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 20:
                total_texts += 1

                # Store
                start = time.perf_counter()
                hash_key, annotated = store.store(
                    original_text=content,
                    compressed_text=content[:len(content)//2],  # simulate 50% compression
                )
                store_latencies.append((time.perf_counter() - start) * 1000)
                stored_hashes.append(hash_key)
                store_success_count += 1

                # Verify marker injection
                assert "[TO:retrieve hash=" in annotated, f"Marker not found in annotated text"

                # Retrieve
                start = time.perf_counter()
                retrieved = store.retrieve(hash_key)
                retrieve_latencies.append((time.perf_counter() - start) * 1000)
                if retrieved == content:
                    retrieve_success_count += 1

    return {
        "store_count": store_success_count,
        "retrieve_success_count": retrieve_success_count,
        "total_texts": total_texts,
        "store_success_rate": round(store_success_count / max(1, total_texts), 4),
        "retrieve_success_rate": round(retrieve_success_count / max(1, total_texts), 4),
        "store_latency_ms": {
            "mean": round(statistics.mean(store_latencies), 4) if store_latencies else 0,
            "std": round(statistics.stdev(store_latencies), 4) if len(store_latencies) > 1 else 0,
        },
        "retrieve_latency_ms": {
            "mean": round(statistics.mean(retrieve_latencies), 4) if retrieve_latencies else 0,
            "std": round(statistics.stdev(retrieve_latencies), 4) if len(retrieve_latencies) > 1 else 0,
        },
        "store_stats": store.stats,
        "hit_rate": round(store.get_hit_rate(), 4),
    }


def benchmark_cache_aligner(messages_a: list[dict], messages_b: list[dict], iterations: int = 5) -> dict:
    """Benchmark CacheAligner (DynamicContentDetector) for prefix stability.

    Tests whether extracting dynamic content from system prompts improves
    the prefix hash stability between consecutive requests.
    """
    detector = DynamicContentDetector()

    # Baseline: standard reorder (no dynamic extraction)
    baseline_stability_scores = []
    baseline_hash_matches = 0

    for _ in range(iterations):
        reordered_a, meta_a = reorder_messages(messages_a, enable_dynamic_extraction=False)
        reordered_b, meta_b = reorder_messages(messages_b, enable_dynamic_extraction=False)

        hash_a = compute_prefix_hash(reordered_a)
        hash_b = compute_prefix_hash(reordered_b)

        if hash_a == hash_b:
            baseline_hash_matches += 1

        stability = detector.get_prefix_stability_score(reordered_a, reordered_b)
        baseline_stability_scores.append(stability)

    # Enhanced: with CacheAligner (enable_dynamic_extraction=True)
    enhanced_stability_scores = []
    enhanced_hash_matches = 0
    dynamic_fields_counts = []

    for _ in range(iterations):
        reordered_a, meta_a = reorder_messages(messages_a, enable_dynamic_extraction=True)
        reordered_b, meta_b = reorder_messages(messages_b, enable_dynamic_extraction=True)

        hash_a = compute_prefix_hash(reordered_a)
        hash_b = compute_prefix_hash(reordered_b)

        if hash_a == hash_b:
            enhanced_hash_matches += 1

        stability = detector.get_prefix_stability_score(reordered_a, reordered_b)
        enhanced_stability_scores.append(stability)
        dynamic_fields_counts.append(meta_b.get("dynamic_fields_extracted", 0))

    return {
        "baseline": {
            "stability_score": {
                "mean": round(statistics.mean(baseline_stability_scores), 4),
                "std": round(statistics.stdev(baseline_stability_scores), 4) if len(baseline_stability_scores) > 1 else 0,
            },
            "prefix_hash_match_rate": round(baseline_hash_matches / iterations, 4),
        },
        "enhanced": {
            "stability_score": {
                "mean": round(statistics.mean(enhanced_stability_scores), 4),
                "std": round(statistics.stdev(enhanced_stability_scores), 4) if len(enhanced_stability_scores) > 1 else 0,
            },
            "prefix_hash_match_rate": round(enhanced_hash_matches / iterations, 4),
            "dynamic_fields_extracted": {
                "mean": round(statistics.mean(dynamic_fields_counts), 1),
                "std": round(statistics.stdev(dynamic_fields_counts), 1) if len(dynamic_fields_counts) > 1 else 0,
            },
        },
    }


def benchmark_statistical_analyzer(messages: list[dict], iterations: int = 5) -> dict:
    """Benchmark StatisticalAnalyzer accuracy and latency.

    Tests field type inference on structured data in messages.
    """
    analyzer = StatisticalAnalyzer()

    analysis_latencies = []
    field_type_results = []
    lossless_attempts = 0
    lossless_successes = 0

    for _ in range(iterations):
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            # Try to find JSON content
            json_start = content.find('{')
            json_end = content.rfind('}')
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end+1]
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    continue

                start = time.perf_counter()
                text_result, analyses = analyzer.analyze_and_compress_text(json_str)
                elapsed = (time.perf_counter() - start) * 1000
                analysis_latencies.append(elapsed)

                for a in analyses:
                    field_type_results.append(a.field_type.value)

                # Try lossless compression
                if isinstance(data, dict) and "data" in data:
                    inner = data["data"]
                    if isinstance(inner, dict) and "metrics" in inner:
                        lossless_attempts += 1
                        result, meta = analyzer.try_lossless_csv_schema(inner["metrics"])
                        if result is not None:
                            lossless_successes += 1

    # Test anomaly detection
    anomaly_test_values = [10, 12, 11, 13, 10, 12, 100, 11, 13, 10]  # 100 is anomaly
    anomaly_result = analyzer.analyze_values(anomaly_test_values)

    # Test array end preservation
    long_array = list(range(100))
    preserved = analyzer.preserve_array_ends(long_array)
    assert len(preserved) == 5, f"Expected 5, got {len(preserved)}"
    assert preserved[0] == 0 and preserved[-1] == 99, "First/last not preserved"

    return {
        "analysis_latency_ms": {
            "mean": round(statistics.mean(analysis_latencies), 4) if analysis_latencies else 0,
            "std": round(statistics.stdev(analysis_latencies), 4) if len(analysis_latencies) > 1 else 0,
        },
        "field_types_detected": list(set(field_type_results)),
        "field_type_count": len(set(field_type_results)),
        "anomaly_detection": {
            "detected": anomaly_result.is_anomaly,
            "expected": True,
            "correct": anomaly_result.is_anomaly == True,
        },
        "array_end_preservation": {
            "input_length": 100,
            "output_length": len(preserved),
            "first_preserved": preserved[0] == 0,
            "last_preserved": preserved[-1] == 99,
        },
        "lossless_compression": {
            "attempts": lossless_attempts,
            "successes": lossless_successes,
            "success_rate": round(lossless_successes / max(1, lossless_attempts), 4),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main Benchmark Runner
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark() -> dict:
    """Run all benchmark scenarios and collect results."""
    print("=" * 60)
    print("Headroom Integration Benchmark")
    print("=" * 60)
    iterations = 5
    results = {
        "benchmark_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "iterations_per_scenario": iterations,
            "scenarios": 6,
            "techniques_tested": [
                "CCR (Compression with Content Recall)",
                "CacheAligner (DynamicContentDetector)",
                "StatisticalAnalyzer (SmartCrusher)",
            ],
        },
        "scenarios": {},
    }

    # Scenario 1: Short conversation
    print("\n[1/6] Short conversation (<100 tokens)...")
    msgs = scenario_short_conversation()
    s1 = {
        "name": "short_conversation",
        "description": "<100 tokens, minimal content",
        "compression": benchmark_compression(msgs, iterations),
        "ccr": benchmark_ccr(msgs, iterations),
        "statistical_analyzer": benchmark_statistical_analyzer(msgs, iterations),
    }
    results["scenarios"]["short_conversation"] = s1
    print(f"  Original tokens: {s1['compression']['original_tokens']}")
    print(f"  Baseline ratio: {s1['compression']['baseline']['compression_ratio']['mean']:.3f}")
    print(f"  Enhanced ratio: {s1['compression']['enhanced']['compression_ratio']['mean']:.3f}")

    # Scenario 2: Medium conversation
    print("\n[2/6] Medium conversation (100-500 tokens)...")
    msgs = scenario_medium_conversation()
    s2 = {
        "name": "medium_conversation",
        "description": "100-500 tokens, with code",
        "compression": benchmark_compression(msgs, iterations),
        "ccr": benchmark_ccr(msgs, iterations),
        "statistical_analyzer": benchmark_statistical_analyzer(msgs, iterations),
    }
    results["scenarios"]["medium_conversation"] = s2
    print(f"  Original tokens: {s2['compression']['original_tokens']}")
    print(f"  Baseline ratio: {s2['compression']['baseline']['compression_ratio']['mean']:.3f}")
    print(f"  Enhanced ratio: {s2['compression']['enhanced']['compression_ratio']['mean']:.3f}")

    # Scenario 3: Long conversation
    print("\n[3/6] Long conversation (500+ tokens)...")
    msgs = scenario_long_conversation()
    s3 = {
        "name": "long_conversation",
        "description": "500+ tokens, detailed architecture discussion",
        "compression": benchmark_compression(msgs, iterations),
        "ccr": benchmark_ccr(msgs, iterations),
        "statistical_analyzer": benchmark_statistical_analyzer(msgs, iterations),
    }
    results["scenarios"]["long_conversation"] = s3
    print(f"  Original tokens: {s3['compression']['original_tokens']}")
    print(f"  Baseline ratio: {s3['compression']['baseline']['compression_ratio']['mean']:.3f}")
    print(f"  Enhanced ratio: {s3['compression']['enhanced']['compression_ratio']['mean']:.3f}")

    # Scenario 4: JSON tool output
    print("\n[4/6] JSON tool output...")
    msgs = scenario_json_tool_output()
    s4 = {
        "name": "json_tool_output",
        "description": "API response with metrics, alerts, and metadata",
        "compression": benchmark_compression(msgs, iterations),
        "ccr": benchmark_ccr(msgs, iterations),
        "statistical_analyzer": benchmark_statistical_analyzer(msgs, iterations),
    }
    results["scenarios"]["json_tool_output"] = s4
    print(f"  Original tokens: {s4['compression']['original_tokens']}")
    print(f"  Baseline ratio: {s4['compression']['baseline']['compression_ratio']['mean']:.3f}")
    print(f"  Enhanced ratio: {s4['compression']['enhanced']['compression_ratio']['mean']:.3f}")
    print(f"  Field types detected: {s4['statistical_analyzer']['field_types_detected']}")

    # Scenario 5: Code snippet
    print("\n[5/6] Code snippet...")
    msgs = scenario_code_snippet()
    s5 = {
        "name": "code_snippet",
        "description": "Python class with type hints, multiple methods",
        "compression": benchmark_compression(msgs, iterations),
        "ccr": benchmark_ccr(msgs, iterations),
        "statistical_analyzer": benchmark_statistical_analyzer(msgs, iterations),
    }
    results["scenarios"]["code_snippet"] = s5
    print(f"  Original tokens: {s5['compression']['original_tokens']}")
    print(f"  Baseline ratio: {s5['compression']['baseline']['compression_ratio']['mean']:.3f}")
    print(f"  Enhanced ratio: {s5['compression']['enhanced']['compression_ratio']['mean']:.3f}")

    # Scenario 6: Dynamic system prompt
    print("\n[6/6] Dynamic system prompt (CacheAligner test)...")
    msgs_a, msgs_b = scenario_dynamic_system_prompt()
    s6 = {
        "name": "dynamic_system_prompt",
        "description": "System prompt with dates, UUIDs, session IDs changing between requests",
        "cache_aligner": benchmark_cache_aligner(msgs_a, msgs_b, iterations),
    }
    results["scenarios"]["dynamic_system_prompt"] = s6
    print(f"  Baseline hash match rate: {s6['cache_aligner']['baseline']['prefix_hash_match_rate']:.1%}")
    print(f"  Enhanced hash match rate: {s6['cache_aligner']['enhanced']['prefix_hash_match_rate']:.1%}")
    print(f"  Baseline stability: {s6['cache_aligner']['baseline']['stability_score']['mean']:.4f}")
    print(f"  Enhanced stability: {s6['cache_aligner']['enhanced']['stability_score']['mean']:.4f}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Aggregate compression improvement
    compression_improvements = []
    for key in ["short_conversation", "medium_conversation", "long_conversation", "json_tool_output", "code_snippet"]:
        s = results["scenarios"][key]
        baseline_ratio = s["compression"]["baseline"]["compression_ratio"]["mean"]
        enhanced_ratio = s["compression"]["enhanced"]["compression_ratio"]["mean"]
        if baseline_ratio > 0:
            improvement = (baseline_ratio - enhanced_ratio) / baseline_ratio * 100
            compression_improvements.append(improvement)

    avg_compression_improvement = statistics.mean(compression_improvements) if compression_improvements else 0

    # CCR success rate
    ccr_rates = []
    for key in ["short_conversation", "medium_conversation", "long_conversation", "json_tool_output", "code_snippet"]:
        rate = results["scenarios"][key]["ccr"]["retrieve_success_rate"]
        ccr_rates.append(rate)
    avg_ccr_rate = statistics.mean(ccr_rates) if ccr_rates else 0

    # CacheAligner improvement
    ca = results["scenarios"]["dynamic_system_prompt"]["cache_aligner"]
    baseline_stability = ca["baseline"]["stability_score"]["mean"]
    enhanced_stability = ca["enhanced"]["stability_score"]["mean"]
    stability_improvement = enhanced_stability - baseline_stability

    results["summary"] = {
        "avg_compression_improvement_pct": round(avg_compression_improvement, 2),
        "avg_ccr_retrieve_success_rate": round(avg_ccr_rate, 4),
        "cache_aligner": {
            "baseline_stability": round(baseline_stability, 4),
            "enhanced_stability": round(enhanced_stability, 4),
            "improvement": round(stability_improvement, 4),
            "baseline_hash_match_rate": ca["baseline"]["prefix_hash_match_rate"],
            "enhanced_hash_match_rate": ca["enhanced"]["prefix_hash_match_rate"],
        },
    }

    print(f"\nCompression: avg {avg_compression_improvement:+.2f}% change")
    print(f"CCR retrieve success rate: {avg_ccr_rate:.1%}")
    print(f"CacheAligner stability: {baseline_stability:.4f} → {enhanced_stability:.4f} ({stability_improvement:+.4f})")
    print(f"CacheAligner hash match: {ca['baseline']['prefix_hash_match_rate']:.1%} → {ca['enhanced']['prefix_hash_match_rate']:.1%}")

    return results


if __name__ == "__main__":
    results = run_benchmark()

    # Save results
    output_path = PROJECT_ROOT / "benchmark_headroom_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to {output_path}")
