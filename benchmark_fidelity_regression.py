"""Regression benchmark for L1 v5 compression fidelity guard.

This benchmark is deterministic and does not call external APIs. It checks that
safe compressor outputs pass while lossy outputs are rejected. The corpus focuses
on high-risk prompt-compression failure modes: code, paths, URLs, API params,
financial numbers, legal/medical facts, multilingual text, JSON, SQL, logs, and
CLI commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

from token_optimizer.core.smart_compressor import SmartCompressor, estimate_tokens_from_messages


@dataclass(frozen=True)
class FidelityCase:
    name: str
    messages: list[dict[str, str]]
    safe: list[dict[str, str]]
    lossy: list[dict[str, str]]
    category: str


def _case(name: str, content: str, safe: str, lossy: str, category: str, repeat: int = 18) -> FidelityCase:
    return FidelityCase(
        name=name,
        category=category,
        messages=[{"role": "user", "content": content * repeat}],
        safe=[{"role": "user", "content": safe}],
        lossy=[{"role": "user", "content": lossy}],
    )


CASES: list[FidelityCase] = [
    _case(
        "code_path_number_url",
        "修复 /app/data/project/main.py 的 parse_price()，错误码 500，金额 ¥19.9，接口 https://api.example.com/v1/prices。",
        "修复 /app/data/project/main.py 的 parse_price()；保留错误码 500、金额 ¥19.9、接口 https://api.example.com/v1/prices。",
        "修复价格解析函数。",
        "code",
    ),
    _case(
        "json_api_params",
        '请求体 {"user_id": 42, "plan": "pro"}，必须保留 endpoint /v1/billing/checkout 和 status=409。',
        '保留 {"user_id": 42, "plan": "pro"}、/v1/billing/checkout、status=409。',
        "保留计费请求参数。",
        "api",
    ),
    _case(
        "python_traceback",
        "Traceback: File /srv/app/worker.py line 87 in run_job() raised TimeoutError after 30s, job_id=abc-123。",
        "保留 Traceback、/srv/app/worker.py、line 87、run_job()、TimeoutError、30s、job_id=abc-123。",
        "任务超时了，修复一下。",
        "logs",
    ),
    _case(
        "sql_migration",
        "迁移 SQL: ALTER TABLE users ADD COLUMN plan_id INT DEFAULT 0; 回滚文件 /migrations/20260611_add_plan.sql。",
        "保留 ALTER TABLE users ADD COLUMN plan_id INT DEFAULT 0 和 /migrations/20260611_add_plan.sql。",
        "处理用户表迁移。",
        "database",
    ),
    _case(
        "cli_command",
        "执行命令 python -m pytest tests/test_smart_compressor.py -q，失败在 test_profit_guard，退出码 1。",
        "保留命令 python -m pytest tests/test_smart_compressor.py -q、test_profit_guard、退出码 1。",
        "测试失败了。",
        "cli",
    ),
    _case(
        "financial_numbers",
        "小米持仓 1810.HK，买入价 18.72，当前价 19.86，止损 17.50，仓位 12%。",
        "保留小米持仓 1810.HK、买入价 18.72、当前价 19.86、止损 17.50、仓位 12%。",
        "总结小米持仓。",
        "finance",
    ),
    _case(
        "legal_clause",
        "合同第 7.2 条：违约金为订单金额的 15%，争议管辖地为北京市朝阳区人民法院。",
        "保留合同第 7.2 条、违约金为订单金额的 15%、争议管辖地北京市朝阳区人民法院。",
        "总结合同违约条款。",
        "legal",
    ),
    _case(
        "medical_dosage",
        "病例记录：阿莫西林 500mg 每日 3 次，过敏史 penicillin allergy，复诊日期 2026-06-18。",
        "保留阿莫西林 500mg、每日 3 次、penicillin allergy、2026-06-18。",
        "总结用药记录。",
        "medical",
    ),
    _case(
        "multilingual_mixed",
        "Translate 'latency budget exceeded' 为中文，保留术语 VAD、ASR、p95=320ms、device=MacBook M2 Max。",
        "保留 latency budget exceeded、VAD、ASR、p95=320ms、device=MacBook M2 Max。",
        "翻译性能问题。",
        "multilingual",
    ),
    _case(
        "env_vars",
        "部署变量 OPENAI_API_BASE=https://api.openai.com/v1，MODEL=mimo-v2.5-pro，TIMEOUT=120。",
        "保留 OPENAI_API_BASE=https://api.openai.com/v1、MODEL=mimo-v2.5-pro、TIMEOUT=120。",
        "总结部署变量。",
        "config",
    ),
    _case(
        "file_paths",
        "读取 /Users/liangliang/ai-town/src/App.jsx 和 /app/data/所有对话/主对话/MEMORY.md，对比差异。",
        "保留 /Users/liangliang/ai-town/src/App.jsx 和 /app/data/所有对话/主对话/MEMORY.md。",
        "读取两个文件对比。",
        "path",
    ),
    _case(
        "http_statuses",
        "接口 POST /v1/chat/completions 返回 429，retry_after=2.5，request_id=req_789。",
        "保留 POST /v1/chat/completions、429、retry_after=2.5、request_id=req_789。",
        "接口限流了。",
        "api",
    ),
    _case(
        "typescript_symbol",
        "TypeScript 报错：Cannot find name 'CompressionPolicy' in src/core/router.ts:144。",
        "保留 CompressionPolicy、src/core/router.ts:144、Cannot find name。",
        "TypeScript 有个类型错误。",
        "code",
    ),
    _case(
        "regex_pattern",
        r"URL 正则 https?://[^\s)\]}>\"，。；：、]+ 不能吞中文标点后的内容。",
        r"保留正则 https?://[^\s)\]}>\"，。；：、]+ 和中文标点要求。",
        "修复 URL 正则。",
        "regex",
    ),
    _case(
        "docker_port",
        "服务监听 0.0.0.0:8080，健康检查 GET /healthz，容器名 token-optimizer-api。",
        "保留 0.0.0.0:8080、GET /healthz、token-optimizer-api。",
        "总结服务健康检查。",
        "ops",
    ),
    _case(
        "uuid_and_hash",
        "任务 trace_id=ee0f1347-4d1f-440d-ac10-c8445577f6f0，commit=a134883，session=7649827550604132671。",
        "保留 trace_id=ee0f1347-4d1f-440d-ac10-c8445577f6f0、commit=a134883、session=7649827550604132671。",
        "记录任务追踪信息。",
        "tracking",
    ),
    _case(
        "pricing_table",
        "价格：mimo-v2.5-pro input $1.00/M output $3.00/M cache $0.20/M；mimo-v2-flash input $0.10/M。",
        "保留 mimo-v2.5-pro $1.00/M、$3.00/M、$0.20/M，以及 mimo-v2-flash $0.10/M。",
        "总结模型价格。",
        "pricing",
    ),
    _case(
        "benchmark_metrics",
        "benchmark: original=2015, compressed=589, token_saved=70.8%, cost_saved=74.1%, fidelity=11/11。",
        "保留 original=2015、compressed=589、70.8%、74.1%、fidelity=11/11。",
        "总结 benchmark 结果。",
        "benchmark",
    ),
    _case(
        "json_nested",
        '{"route":{"selected_candidate":"mimo-v2-flash","candidate_count":2},"profit_guard":{"passed":true,"margin":0.624}}',
        '保留 selected_candidate=mimo-v2-flash、candidate_count=2、passed=true、margin=0.624。',
        "总结路由结果。",
        "json",
    ),
    _case(
        "csv_columns",
        "CSV 列 user_id,plan,cost_usd,token_saved_pct，其中 cost_usd 不能改成 cost。",
        "保留列 user_id、plan、cost_usd、token_saved_pct，且 cost_usd 不能改名。",
        "总结 CSV 列。",
        "tabular",
    ),
    _case(
        "markdown_link",
        "报告链接 [Adapter 横评报告](computer:///app/data/projects/token-optimizer/competitive_adapter_benchmark_report.md)。",
        "保留 computer:///app/data/projects/token-optimizer/competitive_adapter_benchmark_report.md。",
        "保留报告链接。",
        "markdown",
    ),
    _case(
        "model_route",
        "路由 mimo-v2.5-pro -> mimo-v2-flash 是同平台跨代廉价模型，不是同代兄弟模型。",
        "保留 mimo-v2.5-pro -> mimo-v2-flash、同平台跨代廉价模型、不是同代兄弟模型。",
        "总结模型路由。",
        "routing",
    ),
    _case(
        "permission_boundary",
        "删除 /app/data/project/cache 前必须先备份到 /app/data/project/backup-20260611.tar.gz，并等待确认。",
        "保留删除 /app/data/project/cache、备份 /app/data/project/backup-20260611.tar.gz、等待确认。",
        "删除缓存前注意安全。",
        "safety",
    ),
    _case(
        "calendar_time",
        "日程 202606111430 到 202606111530，rrule DAILY interval=1 until=202606301430。",
        "保留 202606111430、202606111530、DAILY、interval=1、until=202606301430。",
        "总结日程规则。",
        "calendar",
    ),
    _case(
        "shell_error",
        "Shell 报错 syntax error near unexpected token `print(\"cannot use the package\")`，发生在 /tmp/tmp44_iiovs.sh line 1。",
        "保留 syntax error、print(\"cannot use the package\")、/tmp/tmp44_iiovs.sh、line 1。",
        "Shell 有语法错误。",
        "shell",
    ),
    _case(
        "unicode_cjk",
        "中文路径 /app/data/所有对话/主对话/用户上传/报价单.xlsx，金额 ￥12,345.67，税率 6%。",
        "保留 /app/data/所有对话/主对话/用户上传/报价单.xlsx、￥12,345.67、6%。",
        "总结报价单。",
        "unicode",
    ),
    _case(
        "product_requirement",
        "PRD 需求：P0 必须支持 shadow mode，P1 支持 ProviderModelProbe，验收指标 p95<300ms。",
        "保留 P0、shadow mode、P1、ProviderModelProbe、p95<300ms。",
        "总结 PRD。",
        "prd",
    ),
    _case(
        "cache_policy",
        "缓存命中率假设 80%，PRO_CACHE_PRICE=0.20，PRO_INPUT_PRICE=1.00，PRO_OUTPUT_PRICE=3.00。",
        "保留 80%、PRO_CACHE_PRICE=0.20、PRO_INPUT_PRICE=1.00、PRO_OUTPUT_PRICE=3.00。",
        "总结缓存策略。",
        "cache",
    ),
    _case(
        "rate_limit_policy",
        "限流策略：max_consecutive_failures=3，circuit_breaker_cooldown=20，disabled_until_call=45。",
        "保留 max_consecutive_failures=3、circuit_breaker_cooldown=20、disabled_until_call=45。",
        "总结限流策略。",
        "reliability",
    ),
    _case(
        "email_address",
        "联系邮箱 support@unfaze.app，BD 邮箱 bd@coze.cn，反馈邮箱 kzfeedback@coze.email。",
        "保留 support@unfaze.app、bd@coze.cn、kzfeedback@coze.email。",
        "总结联系邮箱。",
        "email",
    ),
]


def run_case(sc: SmartCompressor, messages, output):
    with patch.object(sc, "_call_compressor", return_value=output):
        result, meta = sc.compress(messages)
    return result, meta


def _new_compressor() -> SmartCompressor:
    return SmartCompressor(
        main_model="mimo-v2.5-pro",
        api_key="sk-test-key",
        base_url="https://api.xiaomimimo.com/v1",
        min_rule_tokens_for_smart=1,
    )


def main() -> None:
    safe_passed = 0
    lossy_rejected = 0
    categories: dict[str, dict[str, int]] = {}
    print("L1 v5 Fidelity Regression Benchmark")
    print("=" * 104)
    for case in CASES:
        _, safe_meta = run_case(_new_compressor(), case.messages, case.safe)
        _, lossy_meta = run_case(_new_compressor(), case.messages, case.lossy)
        original_tokens = estimate_tokens_from_messages(case.messages)
        safe_tokens = estimate_tokens_from_messages(case.safe)
        safe_ok = safe_meta["mode"] == "smart" and safe_meta["fidelity_guard"]["passed"]
        lossy_ok = lossy_meta["mode"] == "rule_only_fidelity_guard"
        safe_passed += int(safe_ok)
        lossy_rejected += int(lossy_ok)
        bucket = categories.setdefault(case.category, {"total": 0, "safe": 0, "lossy": 0})
        bucket["total"] += 1
        bucket["safe"] += int(safe_ok)
        bucket["lossy"] += int(lossy_ok)
        print(
            f"{case.name:<28} {case.category:<12} original={original_tokens:<5} safe={safe_tokens:<4} "
            f"safe_ok={str(safe_ok):<5} lossy_rejected={str(lossy_ok):<5} "
            f"safe_score={(safe_meta.get('fidelity_guard') or {}).get('score')} "
            f"lossy_score={(lossy_meta.get('fidelity_guard') or {}).get('score')}"
        )
    total = len(CASES)
    print("-" * 104)
    print(f"Safe pass rate:      {safe_passed}/{total}")
    print(f"Lossy reject rate:   {lossy_rejected}/{total}")
    print("Category coverage:")
    for category, stats in sorted(categories.items()):
        print(f"  - {category:<12} safe={stats['safe']}/{stats['total']} lossy={stats['lossy']}/{stats['total']}")
    if safe_passed != total or lossy_rejected != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
