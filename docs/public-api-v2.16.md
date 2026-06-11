# v2.16 Public API + Semantic Assertion Expansion

## Summary

`token-optimizer` v2.16 adds two productization-oriented layers:

1. A stable public compression API for library consumers.
2. Four additional semantic assertion adversarial cases that stress latest values, nested JSON facts, negations, and threshold comparisons.

The project remains a factual-preserving compression engine for caller-selected text. It is not a context manager, memory store, retriever, ranker, or prompt assembler.

## Public API

New exports from `token_optimizer`:

- `compress_text`
- `CompressionResult`
- `CompressionQuality`
- `CompressionStats`

Example:

```python
from token_optimizer import compress_text

result = compress_text(
    "Owner Liang. Deadline Jun 25. Budget is 900 USD. Do not retry HTTP 4xx.",
    mode="safe",
    content_type="memory",
    preserve=["numbers", "dates", "negations", "identifiers"],
)

print(result.compressed)
print(result.stats.original_tokens)
print(result.stats.compressed_tokens)
print(result.quality.miss_summary)
```

### Modes

| Mode | Base keep ratio | Intended use |
|---|---:|---|
| `safe` | 0.95 | critical facts, memory, financial/legal/medical/code-like content |
| `balanced` | 0.75 | default factual compression |
| `aggressive` | 0.50 | low-risk text where higher loss tolerance is acceptable |

If `budget_tokens` is supplied, the API lowers the keep ratio only as needed and clamps it to a safe range.

### Result shape

`CompressionResult` contains:

- `compressed`
- `original_tokens`
- `compressed_tokens`
- `keep_ratio`
- `method`
- `mode`
- `content_type`
- `warnings`
- `quality`
- `stats` property returning `CompressionStats`

`CompressionQuality.miss_summary` is present from day one with stable keys:

```python
{"alias_gap": 0, "true_loss": 0, "assertion_gap": 0}
```

This reserves a stable integration point for future evaluator/guardrail wiring without changing the public return shape.

## Semantic assertion expansion

New adversarial cases:

| Case | Focus |
|---|---|
| `adv_assert_latest_021` | latest corrected value and stale-marker handling |
| `adv_assert_json_path_022` | nested JSON/path facts and invoice totals |
| `adv_assert_negation_023` | independent negative safety policies |
| `adv_assert_threshold_024` | dense threshold comparison conditions |

Latest adversarial result after expansion:

| Metric | Result |
|---|---:|
| Total cases | 24 |
| QA passed | 205/205 |
| QA rate | 100.0% |
| Perfect cases | 24/24 |
| Failed cases | 0 |
| alias_gap | 0 |
| true_loss | 0 |
| assertion_gap | 0 |

## Implementation note

During v2.16, `AdaptiveCompressor` was fixed to call `NearDeduplicator(similarity_threshold=0.85)` instead of the stale `threshold=0.85` argument. This is a compatibility fix for an existing internal constructor mismatch, not a compression-strategy change.

## Validation commands

```bash
PYTHONPATH=src:. python3 -m pytest tests/test_public_api.py tests/test_adversarial_assertions.py tests/test_quality_benchmark_evaluator.py
python3 benchmarks/adversarial/run_adversarial.py --min-rate 1.0
python3 benchmarks/realworld/run_realworld.py
PYTHONPATH=src python3 -m pytest
python3 quality_benchmark.py
python3 -m compileall -q src quality_benchmark.py benchmarks tests
```
