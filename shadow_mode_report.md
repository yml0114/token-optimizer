# Shadow Mode / Telemetry Report

本报告用于上线前 dry-run：不改变真实请求、不调用廉价模型，只记录 v5 SmartCompressor 如果启用会发生什么。

## Summary

- Scenarios: 11
- Would call smart compression: 11/11 (100%)
- Original tokens: 2015
- Rule tokens: 1392
- Estimated smart tokens: 412
- Rule → smart token saved: 70.4%
- Estimated rule cost: $0.00221700
- Estimated smart cost: $0.00091180
- Estimated incremental cost saved: 58.87%

## Policy Distribution

- protected: 8
- extreme: 3

## Fallback Distribution

- none

## Scenario Rows

| Scenario | Smart? | Policy | Rule tokens | Est. smart tokens | Est. savings | Protected spans | Candidate | Fallback |
|---|---:|---|---:|---:|---:|---:|---|---|
| Coding conversation | yes | protected | 205 | 61 | 58.6% | 11 | mimo-v2-flash | - |
| Tool output | yes | extreme | 71 | 21 | 58.94% | 7 | mimo-v2-flash | - |
| Error debugging | yes | protected | 117 | 35 | 57.96% | 10 | mimo-v2-flash | - |
| Long multi-turn | yes | protected | 231 | 69 | 58.86% | 11 | mimo-v2-flash | - |
| English API | yes | protected | 255 | 76 | 58.5% | 25 | mimo-v2-flash | - |
| v3 heavy filler | yes | protected | 53 | 15 | 59.28% | 3 | mimo-v2-flash | - |
| v3 politeness | yes | protected | 29 | 8 | 62.95% | 1 | mimo-v2-flash | - |
| v4 long coding | yes | protected | 278 | 83 | 58.53% | 18 | mimo-v2-flash | - |
| v4 repeated dedup | yes | extreme | 35 | 10 | 59.82% | 4 | mimo-v2-flash | - |
| v4 word-level | yes | protected | 75 | 22 | 59.92% | 11 | mimo-v2-flash | - |
| v4 repeated dedup v2 | yes | extreme | 43 | 12 | 61.34% | 2 | mimo-v2-flash | - |

## Rollout Meaning

- Shadow mode 可以先在线上旁路采样，不改变主链路输入。
- 只记录收益、策略、protected spans、候选模型、fallback reason。
- 当 telemetry 显示收益稳定且 fallback/保护分布合理，再逐步打开真实 smart compression。
