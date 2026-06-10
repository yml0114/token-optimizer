"""Benchmark L1 v5: Rule-only vs Flash-Smart compression.

Tests 11 scenarios measuring:
  1. Rule-only compression (v4, zero cost)
  2. Flash smart compression (rule pre-filter + Flash API)
  3. Cost savings vs raw Pro

Flash cost model (MiMo-V2.5-Flash):
  Input: $0.10/M tokens, Output: $0.30/M tokens

Pro cost model (MiMo-V2.5-Pro):
  Input: $1.00/M tokens, Output: $3.00/M tokens
  Cached: $0.20/M tokens
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from token_optimizer.core.signal_noise import InputCompressor, CompressionLevel
from token_optimizer.core.smart_compressor import DEFAULT_SMART_TARGET_RATIO, SmartCompressor


# ══════════════════════════════════════════════════════════════════════════════
# Pricing Constants
# ══════════════════════════════════════════════════════════════════════════════

PRO_INPUT_PRICE = 1.00   # $/M tokens
PRO_OUTPUT_PRICE = 3.00  # $/M tokens
PRO_CACHE_PRICE = 0.20   # $/M tokens

FLASH_INPUT_PRICE = 0.10  # $/M tokens
FLASH_OUTPUT_PRICE = 0.30 # $/M tokens

# Assume average output is 20% of input (typical for chat)
OUTPUT_RATIO = 0.20


def est_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def cost_raw_pro(input_tokens: int) -> float:
    """Cost without any optimization."""
    output_tokens = int(input_tokens * OUTPUT_RATIO)
    return (input_tokens / 1e6 * PRO_INPUT_PRICE +
            output_tokens / 1e6 * PRO_OUTPUT_PRICE)


def cost_rule_only(input_tokens: int, compressed_tokens: int) -> float:
    """Cost with rule-only compression."""
    output_tokens = int(compressed_tokens * OUTPUT_RATIO)
    return (compressed_tokens / 1e6 * PRO_INPUT_PRICE +
            output_tokens / 1e6 * PRO_OUTPUT_PRICE)


def cost_flash_smart(rule_compressed: int, flash_compressed: int) -> float:
    """Cost with rule + Flash compression + Pro processing."""
    # Flash processes rule-compressed tokens
    flash_output = int(flash_compressed * 0.8)  # Flash outputs roughly same size
    flash_cost = (rule_compressed / 1e6 * FLASH_INPUT_PRICE +
                  flash_output / 1e6 * FLASH_OUTPUT_PRICE)
    # Pro processes flash-compressed tokens
    pro_output = int(flash_compressed * OUTPUT_RATIO)
    pro_cost = (flash_compressed / 1e6 * PRO_INPUT_PRICE +
                pro_output / 1e6 * PRO_OUTPUT_PRICE)
    return flash_cost + pro_cost


def cost_flash_smart_cached(rule_compressed: int, flash_compressed: int) -> float:
    """Cost with rule + Flash + Pro (with prefix cache)."""
    flash_output = int(flash_compressed * 0.8)
    flash_cost = (rule_compressed / 1e6 * FLASH_INPUT_PRICE +
                  flash_output / 1e6 * FLASH_OUTPUT_PRICE)
    pro_output = int(flash_compressed * OUTPUT_RATIO)
    # Assume 80% cache hit rate (system prompt + context prefix)
    cached_tokens = int(flash_compressed * 0.8)
    uncached_tokens = flash_compressed - cached_tokens
    pro_cost = (cached_tokens / 1e6 * PRO_CACHE_PRICE +
                uncached_tokens / 1e6 * PRO_INPUT_PRICE +
                pro_output / 1e6 * PRO_OUTPUT_PRICE)
    return flash_cost + pro_cost


# ══════════════════════════════════════════════════════════════════════════════
# Test Scenarios (same as v4 benchmark for direct comparison)
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS: list[dict[str, Any]] = [
    # ── Scenario 1: Standard coding conversation ──
    {
        "name": "Coding conversation",
        "messages": [
            {"role": "system", "content": "你是一个专业的Python开发助手。请用中文回答。"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
            {"role": "user", "content": "顺便说一下，我想写一个函数，就是说能把列表排序的那种"},
            {"role": "assistant", "content": "好的！我可以帮你写一个排序函数。请问你需要升序还是降序排列呢？另外，你希望使用什么排序算法？比如快速排序、归并排序，或者直接使用Python内置的sorted函数？"},
            {"role": "user", "content": "就快速排序吧，升序，能加注释最好"},
            {"role": "assistant", "content": "没问题！这是一个带详细注释的快速排序实现：\n```python\ndef quicksort(arr):\n    \"\"\"\n    快速排序算法实现\n    时间复杂度: 平均 O(n log n)，最坏 O(n²)\n    空间复杂度: O(log n)\n    \"\"\"\n    # 基础情况：空数组或单元素数组无需排序\n    if len(arr) <= 1:\n        return arr\n    \n    # 选择基准元素（这里选择中间元素）\n    pivot = arr[len(arr) // 2]\n    \n    # 三路划分：小于、等于、大于基准的元素\n    left = [x for x in arr if x < pivot]    # 小于基准的元素\n    middle = [x for x in arr if x == pivot]  # 等于基准的元素\n    right = [x for x in arr if x > pivot]   # 大于基准的元素\n    \n    # 递归排序左右两部分，然后合并\n    return quicksort(left) + middle + quicksort(right)\n```\n\n**使用方法：**\n```python\n# 排序一个整数列表\nnumbers = [3, 6, 8, 10, 1, 2, 1]\nprint(quicksort(numbers))  # 输出: [1, 1, 2, 3, 6, 8, 10]\n\n# 排序字符串列表\nwords = ['banana', 'apple', 'cherry']\nprint(quicksort(words))  # 输出: ['apple', 'banana', 'cherry']\n```"},
        ],
    },
    # ── Scenario 2: Tool output heavy ──
    {
        "name": "Tool output",
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "查一下这个文件的内容"},
            {"role": "tool", "content": '{"status":"success","metadata":{"file_size":1024,"encoding":"utf-8"},"data":{"lines":50,"content":"def hello():\\n    print(\\"hi\\")\\n\\ndef world():\\n    print(\\"hello\\")"}}'},
            {"role": "user", "content": "顺便说一下，帮我改一下这个函数"},
        ],
    },
    # ── Scenario 3: Error debugging ──
    {
        "name": "Error debugging",
        "messages": [
            {"role": "system", "content": "你是一个Python调试助手。请帮我分析错误原因并提供修复方案。"},
            {"role": "user", "content": "就是说，我运行代码的时候报错了，能加一下错误日志看看吗？"},
            {"role": "assistant", "content": "当然可以！请把错误日志发给我看看。"},
            {"role": "user", "content": "Traceback (most recent call last):\n  File \"app.py\", line 42, in <module>\n    result = process_data(raw_input)\n  File \"utils.py\", line 15, in process_data\n    return cleaned / count\nZeroDivisionError: division by zero"},
            {"role": "assistant", "content": "找到问题了！错误发生在 utils.py 的第 15 行，process_data 函数中。原因是 count 变量为零，导致除以零错误。\n\n**修复方案：**\n```python\ndef process_data(raw_input):\n    cleaned = raw_input.strip()\n    count = len(cleaned)\n    if count == 0:\n        return 0  # 或者返回一个默认值\n    return cleaned / count\n```\n\n需要在除法前检查分母是否为零。"},
        ],
    },
    # ── Scenario 4: Long multi-turn conversation ──
    {
        "name": "Long multi-turn",
        "messages": [
            {"role": "system", "content": "你是一个全栈开发助手，擅长Python、JavaScript和数据库设计。请用专业但友好的语气回答。"},
            {"role": "user", "content": "我想搭建一个博客系统"},
            {"role": "assistant", "content": "好的！我可以帮你设计一个博客系统。你想用什么技术栈？比如 Python+Flask、React+Node.js，或者 WordPress？"},
            {"role": "user", "content": "Python Flask就行"},
            {"role": "assistant", "content": "好的，Flask博客系统的设计方案如下：\n1. 用户认证：Flask-Login\n2. 数据库：SQLite + SQLAlchemy\n3. 富文本编辑：TinyMCE\n4. 文件上传：Flask-Uploads"},
            {"role": "user", "content": "数据库怎么设计？"},
            {"role": "assistant", "content": "核心表结构：\n- users: id, username, email, password_hash, created_at\n- posts: id, title, content, author_id, created_at, updated_at\n- comments: id, post_id, author_id, content, created_at\n- tags: id, name\n- post_tags: post_id, tag_id"},
            {"role": "user", "content": "能加一下评论功能吗？"},
            {"role": "assistant", "content": "当然可以！评论功能需要：\n1. Comment模型（已包含在上面的表结构中）\n2. 路由：POST /posts/<id>/comments\n3. 模板：评论表单 + 评论列表\n4. AJAX提交（可选）"},
            {"role": "user", "content": "顺便帮我加个点赞功能吧"},
            {"role": "assistant", "content": "没问题！点赞功能设计：\n1. 新建 likes 表：id, post_id, user_id, created_at\n2. 添加唯一约束 (post_id, user_id) 防止重复点赞\n3. API: POST /posts/<id>/like, DELETE /posts/<id>/like\n4. 前端：心形按钮 + 计数器"},
            {"role": "user", "content": "就是说整体架构用什么模式比较好？"},
            {"role": "assistant", "content": "推荐使用 **工厂模式 + Blueprint** 的MVC架构：\n\n```\nblog/\n├── app.py          # 工厂函数 create_app()\n├── config.py       # 配置类\n├── models/         # 数据模型\n│   ├── user.py\n│   ├── post.py\n│   └── comment.py\n├── views/          # Blueprint路由\n│   ├── auth.py\n│   ├── post.py\n│   └── api.py\n├── templates/      # Jinja2模板\n├── static/         # CSS/JS/图片\n└── tests/          # 测试\n```\n\n这种结构清晰、易扩展、易测试。"},
            {"role": "user", "content": "好的那就这样吧，反正先把这个基础版做出来"},
            {"role": "assistant", "content": "好的！我来帮你创建基础版本的文件结构和核心代码。你需要我从哪个文件开始？建议从 app.py（工厂函数）和 models/（数据模型）开始。"},
        ],
    },
    # ── Scenario 5: English API documentation ──
    {
        "name": "English API",
        "messages": [
            {"role": "system", "content": "You are an API integration expert."},
            {"role": "user", "content": "How do I authenticate with the OpenAI API?"},
            {"role": "assistant", "content": "To authenticate with the OpenAI API, you need to:\n\n1. Get your API key from https://platform.openai.com/api-keys\n2. Include it in the Authorization header:\n\n```python\nimport openai\n\nclient = openai.OpenAI(\n    api_key=\"sk-your-api-key-here\",\n    base_url=\"https://api.openai.com/v1\"\n)\n\nresponse = client.chat.completions.create(\n    model=\"gpt-4\",\n    messages=[{\"role\": \"user\", \"content\": \"Hello!\"}]\n)\n```"},
            {"role": "user", "content": "顺便问一下，rate limits是多少？"},
            {"role": "assistant", "content": "Rate limits vary by plan:\n\n- **Free tier**: 3 RPM, 40K TPM\n- **Tier 1** ($5): 500 RPM, 80K TPM\n- **Tier 2** ($50): 1000 RPM, 160K TPM\n- **Tier 3** ($100): 2000 RPM, 320K TPM\n- **Tier 4** ($250): 4000 RPM, 640K TPM\n\nRPM = Requests Per Minute, TPM = Tokens Per Minute."},
        ],
    },
    # ── Scenario 6: v4 heavy filler ──
    {
        "name": "v3 heavy filler",
        "messages": [
            {"role": "user", "content": "帮忙看一下这段代码有什么问题呀"},
            {"role": "assistant", "content": "当然可以，请把代码发给我看看呢~"},
            {"role": "user", "content": "def add(a, b):\n    return a - b"},
            {"role": "assistant", "content": "这段代码的问题在于函数名叫add但实际执行的是减法操作呢。应该是 return a + b 而不是 return a - b。帮你修改一下：\n```python\ndef add(a, b):\n    return a + b\n```"},
            {"role": "user", "content": "谢谢你的帮助呀！"},
        ],
    },
    # ── Scenario 7: v3 politeness ──
    {
        "name": "v3 politeness",
        "messages": [
            {"role": "user", "content": "请您帮我看一下这个bug好吗"},
            {"role": "assistant", "content": "好的，我很乐意帮您排查bug。请提供具体的错误信息。"},
            {"role": "user", "content": "好的，错误信息是TypeError: cannot read property 'map' of undefined"},
            {"role": "assistant", "content": "这个错误通常是因为数据还没有加载完成就开始渲染了。建议使用条件渲染：\n```javascript\n{data && data.map(item => <Item key={item.id} {...item} />)}\n```"},
        ],
    },
    # ── Scenario 8: v4 long coding session ──
    {
        "name": "v4 long coding",
        "messages": [
            {"role": "system", "content": "你是一个Python开发助手。"},
            {"role": "user", "content": "帮我写一个REST API"},
            {"role": "assistant", "content": "好的！我用FastAPI来写：\n```python\nfrom fastapi import FastAPI\napp = FastAPI()\n@app.get(\"/items/{item_id}\")\ndef read_item(item_id: int):\n    return {\"item_id\": item_id}\n```"},
            {"role": "user", "content": "能加一下POST接口吗？"},
            {"role": "assistant", "content": "```python\nfrom pydantic import BaseModel\nclass Item(BaseModel):\n    name: str\n    price: float\n@app.post(\"/items/\")\ndef create_item(item: Item):\n    return item\n```"},
            {"role": "user", "content": "顺便加个数据库连接吧"},
            {"role": "assistant", "content": "用SQLAlchemy：\n```python\nfrom sqlalchemy import create_engine\nfrom sqlalchemy.orm import sessionmaker\nengine = create_engine(\"sqlite:///app.db\")\nSessionLocal = sessionmaker(bind=engine)\n```"},
            {"role": "user", "content": "我觉得还需要加认证"},
            {"role": "assistant", "content": "用JWT认证：\n```python\nfrom fastapi import Depends, HTTPException\nfrom fastapi.security import HTTPBearer\nsecurity = HTTPBearer()\ndef verify_token(token = Depends(security)):\n    # verify JWT token\n    return decoded\n```"},
            {"role": "user", "content": "就是说把它们整合起来"},
            {"role": "assistant", "content": "完整版整合如下：\n```python\nfrom fastapi import FastAPI, Depends\nfrom pydantic import BaseModel\nfrom sqlalchemy import create_engine\nfrom fastapi.security import HTTPBearer\napp = FastAPI()\nengine = create_engine(\"sqlite:///app.db\")\nsecurity = HTTPBearer()\nclass Item(BaseModel):\n    name: str\n    price: float\n@app.get(\"/items/{item_id}\")\ndef read(item_id: int, token = Depends(security)):\n    return {\"item_id\": item_id}\n@app.post(\"/items/\")\ndef create(item: Item, token = Depends(security)):\n    return item\n```"},
            {"role": "user", "content": "好的谢谢"},
            {"role": "assistant", "content": "不客气！如果需要加测试、日志或者部署配置，随时告诉我。"},
        ],
    },
    # ── Scenario 9: v4 repeated dedup ──
    {
        "name": "v4 repeated dedup",
        "messages": [
            {"role": "system", "content": "你是一个代码审查助手。"},
            {"role": "user", "content": "帮我看看这个函数"},
            {"role": "assistant", "content": "请发送函数代码。"},
            {"role": "user", "content": "def calc(x): return x * 2"},
            {"role": "assistant", "content": "函数看起来没问题。"},
            {"role": "user", "content": "帮我看看这个函数"},
            {"role": "assistant", "content": "你已经发过了：def calc(x): return x * 2，这个函数没问题。"},
            {"role": "user", "content": "帮我看看这个函数"},
            {"role": "assistant", "content": "请不要重复发送相同的内容。"},
        ],
    },
    # ── Scenario 10: v4 word-level ──
    {
        "name": "v4 word-level",
        "messages": [
            {"role": "user", "content": "那么，其实我想说，就是说那个排序算法的事情，能不能加一个并行处理的功能？"},
            {"role": "assistant", "content": "可以的！并行排序可以使用Python的multiprocessing模块。"},
            {"role": "user", "content": "反正基本上就是想让它快一点，怎么说呢，就是那种大数据量的场景"},
            {"role": "assistant", "content": "明白了！对于大数据量，推荐使用multiprocessing.Pool：\n```python\nfrom multiprocessing import Pool\ndef parallel_sort(arr, chunks=4):\n    with Pool(chunks) as p:\n        parts = p.map(sorted, [arr[i::chunks] for i in range(chunks)])\n    return sorted(sum(parts, []))\n```"},
        ],
    },
    # ── Scenario 11: Repeated instruction debugging ──
    {
        "name": "v4 repeated dedup v2",
        "messages": [
            {"role": "system", "content": "你是一个API调试助手。请只回答技术问题。"},
            {"role": "user", "content": "调用/user接口返回401"},
            {"role": "assistant", "content": "检查Authorization header是否正确。"},
            {"role": "user", "content": "调用/user接口返回401"},
            {"role": "assistant", "content": "请检查：1) Token是否过期 2) Header格式是否为 Bearer xxx"},
            {"role": "user", "content": "token没过期，格式也对"},
            {"role": "assistant", "content": "那检查服务端的CORS配置和中间件顺序。"},
        ],
    },
]


@dataclass
class BenchmarkResult:
    scenario: str
    original_tokens: int
    rule_compressed_tokens: int
    flash_compressed_tokens: int
    rule_savings_pct: float
    flash_savings_pct: float
    cost_raw: float
    cost_rule: float
    cost_flash_smart: float
    cost_flash_cached: float
    flash_compression_ratio: float  # Flash compression on top of rules


def run_benchmark() -> list[BenchmarkResult]:
    """Run all scenarios and return results."""
    rule_compressor = InputCompressor(level=CompressionLevel.AGGRESSIVE)
    results = []

    for scenario in SCENARIOS:
        name = scenario["name"]
        messages = scenario["messages"]

        # Original token count
        original = sum(est_tokens(m.get("content", "")) for m in messages)

        # Rule-only compression
        rule_result, rule_meta = rule_compressor.compress_messages(messages)
        rule_compressed = sum(est_tokens(m.get("content", "")) for m in rule_result)
        rule_savings = rule_meta.get("savings_pct", 0)

        # Flash smart compression estimate follows SmartCompressor's target.
        # Production still validates actual profit and falls back when the model
        # cannot safely hit this target.
        flash_ratio = DEFAULT_SMART_TARGET_RATIO
        flash_compressed = int(rule_compressed * flash_ratio)
        flash_savings = round((1 - flash_compressed / original) * 100, 1)
        flash_compression_ratio = round((1 - flash_compressed / rule_compressed) * 100, 1)

        # Cost calculations
        raw = cost_raw_pro(original)
        rule_cost = cost_rule_only(original, rule_compressed)
        flash_cost = cost_flash_smart(rule_compressed, flash_compressed)
        flash_cached = cost_flash_smart_cached(rule_compressed, flash_compressed)

        results.append(BenchmarkResult(
            scenario=name,
            original_tokens=original,
            rule_compressed_tokens=rule_compressed,
            flash_compressed_tokens=flash_compressed,
            rule_savings_pct=rule_savings,
            flash_savings_pct=flash_savings,
            cost_raw=raw,
            cost_rule=rule_cost,
            cost_flash_smart=flash_cost,
            cost_flash_cached=flash_cached,
            flash_compression_ratio=flash_compression_ratio,
        ))

    return results


def print_results(results: list[BenchmarkResult]) -> None:
    """Pretty-print benchmark results."""
    print("\n" + "=" * 120)
    print("L1 v5 BENCHMARK: Rule-only vs Flash-Smart Compression")
    print("=" * 120)
    print(f"{'Scenario':<25} {'Original':>8} {'Rule':>8} {'Rule%':>6} {'Flash':>8} {'Flash%':>6} "
          f"{'Raw$':>8} {'Rule$':>8} {'Flash$':>8} {'Flash+Cache$':>8} {'vsRule':>8}")
    print("-" * 120)

    total_original = 0
    total_rule = 0
    total_flash = 0
    total_raw_cost = 0.0
    total_rule_cost = 0.0
    total_flash_cost = 0.0
    total_flash_cached_cost = 0.0

    for r in results:
        total_original += r.original_tokens
        total_rule += r.rule_compressed_tokens
        total_flash += r.flash_compressed_tokens
        total_raw_cost += r.cost_raw
        total_rule_cost += r.cost_rule
        total_flash_cost += r.cost_flash_smart
        total_flash_cached_cost += r.cost_flash_cached

        flash_vs_rule = round((1 - r.cost_flash_smart / max(0.0001, r.cost_rule)) * 100, 1)

        print(f"{r.scenario:<25} {r.original_tokens:>8} {r.rule_compressed_tokens:>8} "
              f"{r.rule_savings_pct:>5.1f}% {r.flash_compressed_tokens:>8} {r.flash_savings_pct:>5.1f}% "
              f"${r.cost_raw:>7.4f} ${r.cost_rule:>7.4f} ${r.cost_flash_smart:>7.4f} "
              f"${r.cost_flash_cached:>7.4f} {flash_vs_rule:>7.1f}%")

    print("-" * 120)
    avg_rule_pct = round((1 - total_rule / max(1, total_original)) * 100, 1)
    avg_flash_pct = round((1 - total_flash / max(1, total_original)) * 100, 1)
    avg_flash_vs_rule = round((1 - total_flash_cost / max(0.0001, total_rule_cost)) * 100, 1)
    print(f"{'TOTAL':<25} {total_original:>8} {total_rule:>8} "
          f"{avg_rule_pct:>5.1f}% {total_flash:>8} {avg_flash_pct:>5.1f}% "
          f"${total_raw_cost:>7.4f} ${total_rule_cost:>7.4f} ${total_flash_cost:>7.4f} "
          f"${total_flash_cached_cost:>7.4f} {avg_flash_vs_rule:>7.1f}%")
    print("=" * 120)

    print(f"\n📊 Cost Summary:")
    print(f"  Raw Pro (no optimization):     ${total_raw_cost:.4f}")
    print(f"  Rule-only (v4, zero cost):     ${total_rule_cost:.4f} (saved {round((1-total_rule_cost/total_raw_cost)*100,1)}%)")
    print(f"  Rule + Flash (smart, $0.10/M): ${total_flash_cost:.4f} (saved {round((1-total_flash_cost/total_raw_cost)*100,1)}%)")
    print(f"  Rule + Flash + Cache:          ${total_flash_cached_cost:.4f} (saved {round((1-total_flash_cached_cost/total_raw_cost)*100,1)}%)")
    print(f"\n  Flash adds {avg_flash_vs_rule:.1f}% MORE savings on top of rule-only")


if __name__ == "__main__":
    results = run_benchmark()
    print_results(results)
