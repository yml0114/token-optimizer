#!/usr/bin/env python3
"""
Phase 4 竞品对标 Benchmark
Token Optimizer vs LLMLingua vs Selective Context
同考题、同计数器、多场景横评
"""

import sys
import time
import math
import re
import json
from collections import Counter

sys.path.insert(0, "src")

from token_optimizer.core.signal_noise import InputCompressor, CompressionLevel
from token_optimizer.core.compression_store import CompressionStore
from token_optimizer.core.adaptive_compressor import AdaptiveCompressor
from token_optimizer.core.json_aware import JsonAwareCompressor
from token_optimizer.core.near_dedup import NearDeduplicator

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text): return len(enc.encode(text))
except ImportError:
    def count_tokens(text): return max(1, len(text) // 3)

def msgs_tokens(msgs):
    return sum(count_tokens(m.get("content", "")) for m in msgs if isinstance(m.get("content"), str))


# ══════════════════════════════════════════════════════════════
# 测试数据（与 bench_adaptive.py 完全一致）
# ══════════════════════════════════════════════════════════════

def scenario_short():
    return [
        {"role": "system", "content": "你是一个专业的AI助手，擅长回答各种问题。请用简洁明了的语言回答用户的问题，如果不确定就说明不确定。"},
        {"role": "user", "content": "你好，我想了解一下Python的装饰器是什么？"},
        {"role": "assistant", "content": "装饰器是Python的一个语法糖，用@符号标注。它本质上是一个函数，接受函数作为参数，返回一个新函数。常用场景有日志记录、权限校验、缓存等。"},
        {"role": "user", "content": "能给个例子吗？"},
        {"role": "assistant", "content": "当然可以。这是一个简单的计时装饰器：\n```python\nimport time\ndef timer(func):\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        print(f'{func.__name__} 耗时 {time.time()-start:.2f}s')\n        return result\n    return wrapper\n\n@timer\ndef slow_function():\n    time.sleep(1)\n```\n使用@timer装饰器后，调用slow_function()会自动打印执行时间。"},
        {"role": "user", "content": "明白了，那装饰器可以嵌套使用吗？"},
        {"role": "assistant", "content": "可以的！装饰器可以叠加使用。执行顺序是从下往上（从近到远）：\n```python\n@decorator_a\n@decorator_b\ndef func():\n    pass\n```\n等价于`func = decorator_a(decorator_b(func))`。实际开发中常见的组合有@login_required + @cache等。"},
        {"role": "user", "content": "谢谢，还有个问题，类装饰器和函数装饰器有什么区别？"},
        {"role": "assistant", "content": "主要区别：\n1. **函数装饰器**：最常见，接受函数返回函数\n2. **类装饰器**：用类实现__call__方法，可以维护状态\n\n类装饰器的优势是可以保存装饰器的状态（比如记录调用次数），而函数装饰器通常需要闭包或nonlocal来实现。类装饰器的可读性也更好，适合复杂逻辑。"},
        {"role": "user", "content": "好的，我大概理解了，感谢！"},
        {"role": "assistant", "content": "不客气！装饰器是Python中非常实用的特性。建议你多练习几个例子，特别是理解闭包和函数作为对象这两个概念。有其他问题随时问！"},
    ]

def scenario_long():
    return [
        {"role": "system", "content": "你是一个编程助手，专门帮助用户解决Python开发问题。你需要仔细分析问题，给出详细的代码示例，并解释关键概念。如果用户的问题不清楚，请先确认再回答。"},
        {"role": "user", "content": "我想用Python写一个简单的爬虫，爬取网页上的标题和链接"},
        {"role": "assistant", "content": "好的，我来帮你写一个基础爬虫。首先你需要安装requests和beautifulsoup4这两个库。基本思路是：1. 用requests获取网页HTML 2. 用BeautifulSoup解析HTML 3. 提取标题和链接。下面是一个简单示例：\n```python\nimport requests\nfrom bs4 import BeautifulSoup\n\ndef crawl(url):\n    resp = requests.get(url)\n    soup = BeautifulSoup(resp.text, 'html.parser')\n    titles = [h1.text for h1 in soup.find_all('h1')]\n    links = [a['href'] for a in soup.find_all('a', href=True)]\n    return titles, links\n```"},
        {"role": "user", "content": "怎么处理反爬虫机制？比如有些网站会检测User-Agent"},
        {"role": "assistant", "content": "反爬虫处理主要有几种方式：\n1. **设置请求头**：模拟浏览器访问，特别是User-Agent\n2. **添加延迟**：time.sleep()控制请求频率\n3. **使用代理**：轮换IP地址\n4. **处理Cookies**：保持会话状态\n\n最简单的方式是设置User-Agent：\n```python\nheaders = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}\nresp = requests.get(url, headers=headers)\n```"},
        {"role": "user", "content": "那如果网站用JavaScript动态加载内容怎么办？"},
        {"role": "assistant", "content": "动态加载的内容需要不同的方案：\n1. **Selenium**：模拟真实浏览器，可以执行JS\n2. **Playwright**：微软出品，比Selenium更快更稳定\n3. **分析API**：用开发者工具找真实的API接口，直接调用JSON\n\n推荐先分析API，因为最快最稳定。如果必须用浏览器，Playwright是更好的选择。"},
        {"role": "user", "content": "能给个Playwright的例子吗？"},
        {"role": "assistant", "content": "当然！Playwright示例：\n```python\nfrom playwright.sync_api import sync_playwright\n\ndef crawl_dynamic(url):\n    with sync_playwright() as p:\n        browser = p.chromium.launch(headless=True)\n        page = browser.new_page()\n        page.goto(url)\n        page.wait_for_selector('h1')\n        titles = page.eval_on_selector_all('h1', 'els => els.map(e => e.textContent)')\n        links = page.eval_on_selector_all('a[href]', 'els => els.map(e => e.href)')\n        browser.close()\n        return titles, links\n```\n需要先安装：`pip install playwright && playwright install`"},
        {"role": "user", "content": "我按照你的方法试了，但是遇到了TimeoutError，怎么处理？"},
        {"role": "assistant", "content": "TimeoutError通常有几个原因：\n1. 网络问题：检查网络连接\n2. 选择器错误：元素可能没加载完\n3. 页面结构变化：CSS选择器失效\n\n解决方案：\n```python\npage.goto(url, timeout=30000)  # 30秒\npage.wait_for_load_state('networkidle')\nimport time\nfor i in range(3):\n    try:\n        page.goto(url, timeout=10000)\n        break\n    except TimeoutError:\n        time.sleep(2)\n```"},
        {"role": "user", "content": "明白了。还有个问题，爬取的数据怎么保存比较好？"},
        {"role": "assistant", "content": "数据保存有几种常见方式：\n1. **CSV文件**：简单，适合结构化数据\n2. **JSON文件**：保留数据结构\n3. **SQLite数据库**：适合大量数据，支持查询\n4. **MongoDB**：适合非结构化数据\n\n对于爬虫数据，推荐SQLite：\n```python\nimport sqlite3\ndef save_to_db(data):\n    conn = sqlite3.connect('crawler.db')\n    c = conn.cursor()\n    c.execute('CREATE TABLE IF NOT EXISTS pages (title TEXT, url TEXT)')\n    c.executemany('INSERT INTO pages VALUES (?, ?)', data)\n    conn.commit()\n    conn.close()\n```"},
        {"role": "user", "content": "好的，我想实现一个完整的爬虫系统，包括爬取、解析、存储、去重"},
        {"role": "assistant", "content": "完整的爬虫系统架构：\n```python\nclass WebCrawler:\n    def __init__(self):\n        self.visited = set()\n        self.db = sqlite3.connect('crawler.db')\n    def crawl(self, urls):\n        for url in urls:\n            if url in self.visited: continue\n            self.visited.add(url)\n            html = self.fetch(url)\n            data = self.parse(html)\n            self.save(data)\n```\n关键点：URL去重用set、用队列管理待爬URL、异常处理保证稳定性、日志记录便于调试"},
        {"role": "user", "content": "我想用异步来提高效率，怎么实现？"},
        {"role": "assistant", "content": "异步爬虫用aiohttp + asyncio：\n```python\nimport aiohttp, asyncio\nasync def fetch(session, url):\n    async with session.get(url) as resp:\n        return await resp.text()\nasync def crawl_all(urls):\n    async with aiohttp.ClientSession() as session:\n        tasks = [fetch(session, url) for url in urls]\n        return await asyncio.gather(*tasks)\n```\n注意控制并发数，避免被封：\n```python\nsem = asyncio.Semaphore(10)\nasync def limited_fetch(session, url):\n    async with sem:\n        return await fetch(session, url)\n```"},
        {"role": "user", "content": "爬虫需要遵守什么规则吗？比如robots.txt"},
        {"role": "assistant", "content": "爬虫伦理和法律要求：\n1. **robots.txt**：网站的爬虫协议，应该遵守\n2. **请求频率**：不要过于频繁\n3. **版权问题**：爬取的内容可能有版权\n4. **隐私保护**：不要爬取个人隐私信息\n\n```python\nfrom urllib.robotparser import RobotFileParser\nrp = RobotFileParser()\nrp.set_url('https://example.com/robots.txt')\nrp.read()\ncan_fetch = rp.can_fetch('*', url)\n```"},
        {"role": "user", "content": "我想爬取一个电商网站的商品信息，包括价格、标题、评分"},
        {"role": "assistant", "content": "电商网站爬虫要注意：\n1. **反爬更严格**：通常有验证码、IP限制\n2. **数据结构复杂**：价格可能在JS变量中\n3. **动态加载**：商品列表通常分页加载\n\n建议先分析API看有没有JSON接口，直接调API效率更高。如果必须用浏览器，注意处理懒加载和无限滚动。"},
        {"role": "user", "content": "我实现了基本功能，但是爬取速度很慢，只有每秒2-3个页面"},
        {"role": "assistant", "content": "性能优化建议：\n1. **并发数**：从10增加到50-100\n2. **连接池**：复用TCP连接\n3. **去重优化**：用布隆过滤器替代set\n4. **解析优化**：用lxml替代html.parser\n5. **缓存**：已爬URL不重复请求\n\n目标应该是每秒20-50个页面。"},
        {"role": "user", "content": "有什么好的爬虫框架推荐吗？"},
        {"role": "assistant", "content": "主流爬虫框架：\n1. **Scrapy**：最成熟，功能最全\n2. **PySpider**：Web界面，适合入门\n3. **Crawlee**：新框架，现代化设计\n\n推荐Scrapy，它内置了请求队列和去重、中间件（代理、UA轮换）、数据管道（存储）、分布式支持（Scrapy-Redis）。"},
        {"role": "user", "content": "我用Scrapy的时候遇到了问题，settings.py怎么配置？"},
        {"role": "assistant", "content": "Scrapy核心配置：\n```python\nBOT_NAME = 'my_spider'\nROBOTSTXT_OBEY = True\nCONCURRENT_REQUESTS = 16\nDOWNLOAD_DELAY = 0.5\nCOOKIES_ENABLED = False\nDOWNLOADER_MIDDLEWARES = {\n    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,\n    'myproject.middlewares.RandomUserAgentMiddleware': 400,\n}\nITEM_PIPELINES = {\n    'myproject.pipelines.MongoPipeline': 300,\n}\n```"},
        {"role": "user", "content": "谢谢，我已经实现了基本功能，现在想添加一个监控系统"},
        {"role": "assistant", "content": "爬虫监控系统设计：\n1. 指标采集：爬取速度、成功率、错误类型分布、代理IP质量\n2. 告警规则：成功率低于90%告警、速度下降50%告警、连续失败5次暂停\n3. 技术选型：Prometheus + Grafana（推荐）或日志 + ELK\n\n```python\nfrom prometheus_client import Counter, Gauge\npages_crawled = Counter('pages_crawled', 'Total pages crawled')\nsuccess_rate = Gauge('crawl_success_rate', 'Success rate')\n```"},
        {"role": "user", "content": "我还有一个问题，怎么处理验证码？"},
        {"role": "assistant", "content": "验证码处理方案（按难度）：\n1. 简单验证码：OCR识别（Tesseract/百度API）\n2. 滑块验证码：Selenium模拟拖动\n3. 图文验证码：打码平台（2Captcha等）\n4. reCAPTCHA：几乎无法自动化，建议绕过\n\n最佳实践：优先避免触发验证码（控制频率、模拟真实行为）"},
    ]

def scenario_json():
    return [
        {"role": "system", "content": "你是一个数据分析助手。"},
        {"role": "user", "content": "请帮我分析这份销售数据"},
        {"role": "assistant", "content": "好的，请提供数据。"},
        {"role": "user", "content": json.dumps({
            "sales_report": {
                "period": "2024-Q1",
                "regions": [
                    {"name": "华东", "revenue": 1250000, "growth": 12.5, "products": [
                        {"name": "产品A", "units": 3200, "price": 199, "category": "电子"},
                        {"name": "产品B", "units": 1800, "price": 299, "category": "家居"},
                        {"name": "产品C", "units": 950, "price": 499, "category": "电子"},
                    ]},
                    {"name": "华南", "revenue": 980000, "growth": 8.3, "products": [
                        {"name": "产品A", "units": 2100, "price": 199, "category": "电子"},
                        {"name": "产品D", "units": 1500, "price": 159, "category": "食品"},
                        {"name": "产品E", "units": 800, "price": 399, "category": "家居"},
                    ]},
                    {"name": "华北", "revenue": 750000, "growth": -2.1, "products": [
                        {"name": "产品A", "units": 1800, "price": 199, "category": "电子"},
                        {"name": "产品F", "units": 600, "price": 599, "category": "电子"},
                        {"name": "产品B", "units": 900, "price": 299, "category": "家居"},
                    ]},
                ],
                "summary": {"total_revenue": 2980000, "avg_growth": 6.2, "top_product": "产品A"}
            }
        }, ensure_ascii=False)},
        {"role": "assistant", "content": "数据分析结果：\n1. 华东地区收入最高（125万），增长12.5%\n2. 产品A是三个地区的共同热销产品\n3. 华北地区增长为负（-2.1%），需要关注\n4. 电子类占总收入约60%，家居约25%"},
        {"role": "user", "content": "能深入分析一下产品A在各地区的表现吗？"},
        {"role": "assistant", "content": "产品A跨地区分析：\n- 华东：3200台 × ¥199 = ¥63.68万（占华东总收入50.9%）\n- 华南：2100台 × ¥199 = ¥41.79万（占华南总收入42.6%）\n- 华北：1800台 × ¥199 = ¥35.82万（占华北总收入47.8%）\n\n总销售额¥141.29万，占全公司47.4%。产品A是绝对的主力产品。"},
    ]

def scenario_with_dedup():
    msgs = [
        {"role": "system", "content": "你是一个专业的AI助手，擅长回答各种问题。请用简洁明了的语言回答用户的问题，如果不确定就说明不确定。"},
    ]
    cities = [
        ("北京", 22, "晴", 45, "东北风3级", 68, "多云", "18-25", "小雨", "15-22"),
        ("上海", 26, "多云", 72, "东南风2级", 85, "阴", "23-29", "阵雨", "21-27"),
        ("广州", 30, "晴", 80, "南风2级", 72, "多云", "27-33", "雷阵雨", "26-31"),
        ("深圳", 29, "晴", 78, "南风2级", 65, "多云", "26-32", "雷阵雨", "25-30"),
        ("成都", 20, "阴", 65, "微风", 92, "小雨", "17-23", "阴", "15-20"),
        ("杭州", 24, "多云", 68, "东风2级", 58, "晴", "20-27", "多云", "19-25"),
        ("西安", 23, "晴", 40, "西风2级", 105, "多云", "19-26", "晴", "17-24"),
    ]
    for city, temp, weather, humidity, wind, aqi, fc1w, fc1t, fc2w, fc2t in cities:
        msgs.append({"role": "user", "content": f"查询{city}天气"})
        msgs.append({"role": "tool", "content": json.dumps({"location": city, "temp": temp, "weather": weather, "humidity": humidity, "wind": wind, "aqi": aqi, "forecast": [{"day": "明天", "temp": fc1t, "weather": fc1w}, {"day": "后天", "temp": fc2t, "weather": fc2w}]}, ensure_ascii=False)})
        aqi_level = "优" if aqi <= 50 else ("良" if aqi <= 75 else ("一般" if aqi <= 100 else "轻度污染"))
        msgs.append({"role": "assistant", "content": f"{city}当前天气：{temp}°C，{weather}，{wind}，湿度{humidity}%，空气质量{aqi_level}（AQI {aqi}）。明天{fc1w}{fc1t}°C，后天{fc2w}{fc2t}°C。"})
    return msgs


# ══════════════════════════════════════════════════════════════
# 压缩方案
# ══════════════════════════════════════════════════════════════

# ── 1. Token Optimizer P3 ──
def run_token_optimizer_p3(msgs, runs=3):
    times, toks = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        dedup = NearDeduplicator()
        deduped, _ = dedup.deduplicate(msgs)
        comp = AdaptiveCompressor(json_aware=True, near_dedup=False)  # dedup 已单独做
        compressed, stats = comp.compress(messages=deduped, keep_ratio=0.4)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        toks.append(msgs_tokens(compressed))
    return sum(toks)/len(toks), sum(times)/len(times), stats


# ── 2. LLMLingua-2 ──
_llmlingua_compressor = None

def get_llmlingua():
    global _llmlingua_compressor
    if _llmlingua_compressor is None:
        print("    ⏳ 首次加载 LLMLingua-2 模型（下载+初始化可能需要几分钟）...")
        from llmlingua import PromptCompressor
        _llmlingua_compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            device_map="cpu",
            use_llmlingua2=True,
        )
        print("    ✅ LLMLingua-2 模型加载完成")
    return _llmlingua_compressor

def run_llmlingua(msgs, rate=0.5, runs=3):
    comp = get_llmlingua()
    full_text = "\n".join(f"[{m['role']}] {m['content']}" for m in msgs if isinstance(m.get('content'), str))
    times, toks = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            result = comp.compress_prompt(full_text, rate=rate, force_tokens=['[', ']'])
            compressed = result.get("compressed_prompt", full_text)
        except Exception as e:
            print(f"    ⚠️ LLMLingua error: {e}")
            compressed = full_text
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        toks.append(count_tokens(compressed))
    return sum(toks)/len(toks), sum(times)/len(times)


# ── 3. Selective Context（自信息量过滤） ──
def selective_context_compress(msgs, keep_ratio=0.5):
    t0 = time.perf_counter()
    all_words = []
    for m in msgs:
        if isinstance(m.get("content"), str):
            all_words.extend(re.findall(r'[\w\u4e00-\u9fff]+', m["content"].lower()))
    freq = Counter(all_words)
    total = max(1, sum(freq.values()))

    compressed = []
    for m in msgs:
        content = m.get("content", "")
        if not isinstance(content, str):
            compressed.append(m)
            continue
        sentences = re.split(r'(?<=[。！？.!?\n])', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            compressed.append(m)
            continue
        scored = []
        for sent in sentences:
            words = re.findall(r'[\w\u4e00-\u9fff]+', sent.lower())
            if not words:
                scored.append((sent, 0.0))
                continue
            info = sum(-math.log(max(freq.get(w, 1) / total, 1e-10)) for w in words)
            scored.append((sent, info / len(words)))
        n_keep = max(1, int(len(scored) * keep_ratio))
        sorted_sents = sorted(scored, key=lambda x: x[1], reverse=True)
        kept_indices = set()
        for sent, score in sorted_sents[:n_keep]:
            for i, (s, _) in enumerate(scored):
                if s == sent and i not in kept_indices:
                    kept_indices.add(i)
                    break
        new_content = "".join(scored[i][0] for i in sorted(kept_indices))
        compressed.append({**m, "content": new_content})
    elapsed = (time.perf_counter() - t0) * 1000
    return compressed, elapsed


# ── 4. Random Drop 基线 ──
def random_drop_compress(msgs, drop_ratio=0.3):
    import random; random.seed(42)
    t0 = time.perf_counter()
    compressed = []
    for m in msgs:
        content = m.get("content", "")
        if not isinstance(content, str):
            compressed.append(m)
            continue
        sentences = re.split(r'(?<=[。！？.!?\n])', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        kept = [s for s in sentences if random.random() > drop_ratio]
        if not kept:
            kept = sentences[:1] if sentences else [""]
        compressed.append({**m, "content": "".join(kept)})
    elapsed = (time.perf_counter() - t0) * 1000
    return compressed, elapsed


# ══════════════════════════════════════════════════════════════
# 主测试
# ══════════════════════════════════════════════════════════════

def run_bench(msgs, name, runs=3, llmlingua_rate=0.5, sc_ratio=0.5):
    print(f"\n{'='*90}")
    print(f"📊 {name} | {len(msgs)}条消息 | 原始 {msgs_tokens(msgs)} tokens")
    print(f"{'='*90}")

    orig = msgs_tokens(msgs)
    results = []

    # 1. Token Optimizer P3
    print("  [1/4] Token Optimizer P3...")
    tok, ms, stats = run_token_optimizer_p3(msgs, runs=runs)
    dedup_n = stats.get('near_dedup_merged', 0)
    results.append(("Token Optimizer P3", tok, ms, f"去重{dedup_n}+JA+CCR"))

    # 2. LLMLingua-2
    print("  [2/4] LLMLingua-2...")
    try:
        tok, ms = run_llmlingua(msgs, rate=llmlingua_rate, runs=runs)
        results.append(("LLMLingua-2", tok, ms, f"rate={llmlingua_rate}"))
    except Exception as e:
        print(f"    ⚠️ LLMLingua-2 不可用: {e}")
        results.append(("LLMLingua-2", orig, 0, "不可用"))

    # 3. Selective Context
    print("  [3/4] Selective Context...")
    tok_list, ms_list = [], []
    for _ in range(runs):
        cm, ms = selective_context_compress(msgs, keep_ratio=sc_ratio)
        tok_list.append(msgs_tokens(cm))
        ms_list.append(ms)
    results.append(("Selective Context", sum(tok_list)/len(tok_list), sum(ms_list)/len(ms_list), f"ratio={sc_ratio}"))

    # 4. Random Drop
    print("  [4/4] Random Drop...")
    cm, ms = random_drop_compress(msgs, drop_ratio=0.3)
    results.append(("Random Drop(30%)", msgs_tokens(cm), ms, "基线"))

    # 排序
    results.sort(key=lambda x: x[1])

    print(f"\n{'算法':<25} {'压缩后':>7} {'压缩比':>8} {'延迟':>8} 备注")
    print("─" * 80)
    for tool, tok, ms, note in results:
        saved = (1 - tok / orig) * 100 if orig > 0 else 0
        bar = "█" * int(max(0, saved) / 5)
        print(f"  {tool:<23} {tok:>6.0f}   {saved:>+5.1f}%  {ms:>6.1f}ms  {note}  {bar}")

    return results


def main():
    print("=" * 90)
    print("🏁 Phase 4 竞品对标 Benchmark")
    print("   Token Optimizer vs LLMLingua-2 vs Selective Context")
    print("   同考题、同计数器（tiktoken）、多次均值")
    print("=" * 90)

    all_results = {}
    scenarios = [
        (scenario_short, "短对话", 0.5, 0.5),
        (scenario_long, "长对话", 0.5, 0.5),
        (scenario_json, "JSON", 0.5, 0.7),
        (scenario_with_dedup, "含副本", 0.5, 0.5),
    ]
    for scenario_fn, name, lrate, srate in scenarios:
        all_results[name] = run_bench(scenario_fn(), name, runs=3, llmlingua_rate=lrate, sc_ratio=srate)

    # ── 总结表 ──
    print(f"\n{'='*90}")
    print("📋 总结对比")
    print(f"{'='*90}")
    print(f"\n{'场景':<10} {'Token Opt P3':>14} {'LLMLingua-2':>14} {'SelCtx':>14} {'Random':>14}")
    print("─" * 70)
    for scene, results in all_results.items():
        row = {}
        for tool, tok, ms, note in results:
            row[tool] = f"{tok:.0f}t/{ms:.0f}ms"
        print(f"  {scene:<8} {row.get('Token Optimizer P3','?'):>14} {row.get('LLMLingua-2','?'):>14} {row.get('Selective Context','?'):>14} {row.get('Random Drop(30%)','?'):>14}")


if __name__ == "__main__":
    main()
