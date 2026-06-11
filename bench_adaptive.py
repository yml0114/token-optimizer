#!/usr/bin/env python3
"""Benchmark: Adaptive vs IC+CCR vs 竞品 — 验证自适应路由效果"""

import sys
import time
import re
from collections import Counter

sys.path.insert(0, "src")

from token_optimizer.core.signal_noise import InputCompressor, CompressionLevel
from token_optimizer.core.compression_store import CompressionStore
from token_optimizer.core.adaptive_compressor import AdaptiveCompressor

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text): return len(enc.encode(text))
except ImportError:
    def count_tokens(text): return max(1, len(text) // 3)

def msgs_tokens(msgs):
    return sum(count_tokens(m.get("content", "")) for m in msgs if isinstance(m.get("content", ""), str))

# ══════════════════════════════════════════════════════════════
# 测试数据（与之前横评完全一致）
# ══════════════════════════════════════════════════════════════

def scenario_short():
    """短对话：5轮，典型的Agent交互"""
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
    """长对话：20轮，包含重复指令、工具调用、冗长回复"""
    msgs = [
        {"role": "system", "content": "你是一个编程助手，专门帮助用户解决Python开发问题。你需要仔细分析问题，给出详细的代码示例，并解释关键概念。如果用户的问题不清楚，请先确认再回答。"},
        {"role": "user", "content": "我想用Python写一个简单的爬虫，爬取网页上的标题和链接"},
        {"role": "assistant", "content": "好的，我来帮你写一个基础爬虫。首先你需要安装requests和beautifulsoup4这两个库。基本思路是：1. 用requests获取网页HTML 2. 用BeautifulSoup解析HTML 3. 提取标题和链接。下面是一个简单示例：\n```python\nimport requests\nfrom bs4 import BeautifulSoup\n\ndef crawl(url):\n    resp = requests.get(url)\n    soup = BeautifulSoup(resp.text, 'html.parser')\n    titles = [h1.text for h1 in soup.find_all('h1')]\n    links = [a['href'] for a in soup.find_all('a', href=True)]\n    return titles, links\n```"},
        {"role": "user", "content": "怎么处理反爬虫机制？比如有些网站会检测User-Agent"},
        {"role": "assistant", "content": "反爬虫处理主要有几种方式：\n1. **设置请求头**：模拟浏览器访问，特别是User-Agent\n2. **添加延迟**：time.sleep()控制请求频率\n3. **使用代理**：轮换IP地址\n4. **处理Cookies**：保持会话状态\n\n最简单的方式是设置User-Agent：\n```python\nheaders = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}\nresp = requests.get(url, headers=headers)\n```"},
        {"role": "user", "content": "那如果网站用JavaScript动态加载内容怎么办？"},
        {"role": "assistant", "content": "动态加载的内容需要不同的方案：\n1. **Selenium**：模拟真实浏览器，可以执行JS\n2. **Playwright**：微软出品，比Selenium更快更稳定\n3. **分析API**：用开发者工具找真实的API接口，直接调用JSON\n\n推荐先分析API，因为最快最稳定。如果必须用浏览器，Playwright是更好的选择。"},
        {"role": "user", "content": "能给个Playwright的例子吗？"},
        {"role": "assistant", "content": "当然！Playwright示例：\n```python\nfrom playwright.sync_api import sync_playwright\n\ndef crawl_dynamic(url):\n    with sync_playwright() as p:\n        browser = p.chromium.launch(headless=True)\n        page = browser.new_page()\n        page.goto(url)\n        page.wait_for_selector('h1')  # 等待元素加载\n        titles = page.eval_on_selector_all('h1', 'els => els.map(e => e.textContent)')\n        links = page.eval_on_selector_all('a[href]', 'els => els.map(e => e.href)')\n        browser.close()\n        return titles, links\n```\n需要先安装：`pip install playwright && playwright install`"},
        {"role": "user", "content": "我按照你的方法试了，但是遇到了TimeoutError，怎么处理？"},
        {"role": "assistant", "content": "TimeoutError通常有几个原因：\n1. 网络问题：检查网络连接\n2. 选择器错误：元素可能没加载完\n3. 页面结构变化：CSS选择器失效\n\n解决方案：\n```python\n# 增加超时时间\npage.goto(url, timeout=30000)  # 30秒\n\n# 使用更灵活的等待\npage.wait_for_load_state('networkidle')  # 等待网络空闲\n\n# 添加重试机制\nimport time\nfor i in range(3):\n    try:\n        page.goto(url, timeout=10000)\n        break\n    except TimeoutError:\n        time.sleep(2)\n```"},
        {"role": "user", "content": "明白了。还有个问题，爬取的数据怎么保存比较好？"},
        {"role": "assistant", "content": "数据保存有几种常见方式：\n1. **CSV文件**：简单，适合结构化数据\n2. **JSON文件**：保留数据结构\n3. **SQLite数据库**：适合大量数据，支持查询\n4. **MongoDB**：适合非结构化数据\n\n对于爬虫数据，推荐SQLite：\n```python\nimport sqlite3\n\ndef save_to_db(data):\n    conn = sqlite3.connect('crawler.db')\n    c = conn.cursor()\n    c.execute('CREATE TABLE IF NOT EXISTS pages (title TEXT, url TEXT)')\n    c.executemany('INSERT INTO pages VALUES (?, ?)', data)\n    conn.commit()\n    conn.close()\n```"},
        {"role": "user", "content": "好的，我想实现一个完整的爬虫系统，包括爬取、解析、存储、去重"},
        {"role": "assistant", "content": "完整的爬虫系统架构：\n\n```python\nclass WebCrawler:\n    def __init__(self):\n        self.visited = set()  # URL去重\n        self.db = sqlite3.connect('crawler.db')\n    \n    def crawl(self, urls):\n        for url in urls:\n            if url in self.visited:\n                continue\n            self.visited.add(url)\n            html = self.fetch(url)\n            data = self.parse(html)\n            self.save(data)\n    \n    def fetch(self, url): ...\n    def parse(self, html): ...\n    def save(self, data): ...\n```\n\n关键点：\n- URL去重用set\n- 用队列管理待爬URL\n- 异常处理保证稳定性\n- 日志记录便于调试"},
        {"role": "user", "content": "我想用异步来提高效率，怎么实现？"},
        {"role": "assistant", "content": "异步爬虫用aiohttp + asyncio：\n```python\nimport aiohttp\nimport asyncio\n\nasync def fetch(session, url):\n    async with session.get(url) as resp:\n        return await resp.text()\n\nasync def crawl_all(urls):\n    async with aiohttp.ClientSession() as session:\n        tasks = [fetch(session, url) for url in urls]\n        return await asyncio.gather(*tasks)\n\n# 使用\nresults = asyncio.run(crawl_all(urls))\n```\n\n注意控制并发数，避免被封：\n```python\nsem = asyncio.Semaphore(10)  # 最多10个并发\nasync def limited_fetch(session, url):\n    async with sem:\n        return await fetch(session, url)\n```"},
        {"role": "user", "content": "爬虫需要遵守什么规则吗？比如robots.txt"},
        {"role": "assistant", "content": "爬虫伦理和法律要求：\n1. **robots.txt**：网站的爬虫协议，应该遵守\n2. **请求频率**：不要过于频繁，给服务器压力\n3. **版权问题**：爬取的内容可能有版权\n4. **隐私保护**：不要爬取个人隐私信息\n5. **服务条款**：有些网站明确禁止爬虫\n\n技术上可以用robotparser检查：\n```python\nfrom urllib.robotparser import RobotFileParser\nrp = RobotFileParser()\nrp.set_url('https://example.com/robots.txt')\nrp.read()\ncan_fetch = rp.can_fetch('*', url)\n```"},
        {"role": "user", "content": "我想爬取一个电商网站的商品信息，包括价格、标题、评分"},
        {"role": "assistant", "content": "电商网站爬虫要注意：\n1. **反爬更严格**：通常有验证码、IP限制\n2. **数据结构复杂**：价格可能在JS变量中\n3. **动态加载**：商品列表通常分页加载\n\n建议方案：\n```python\n# 先分析API，看有没有JSON接口\n# 如果有，直接调API效率更高\nimport requests\n\napi_url = 'https://api.example.com/products?page=1'\nheaders = {'Authorization': 'Bearer xxx'}\ndata = requests.get(api_url, headers=headers).json()\n\nfor product in data['products']:\n    print(product['title'], product['price'], product['rating'])\n```\n\n如果必须用浏览器，注意处理懒加载和无限滚动。"},
        {"role": "user", "content": "我实现了基本功能，但是爬取速度很慢，只有每秒2-3个页面"},
        {"role": "assistant", "content": "性能优化建议：\n1. **并发数**：从10增加到50-100（需要代理支持）\n2. **连接池**：复用TCP连接\n3. **去重优化**：用布隆过滤器替代set\n4. **解析优化**：用lxml替代html.parser\n5. **缓存**：已爬URL不重复请求\n\n```python\n# 连接池优化\nconnector = aiohttp.TCPConnector(limit=100, limit_per_host=10)\nsession = aiohttp.ClientSession(connector=connector)\n\n# 布隆过滤器\nfrom pybloom_live import BloomFilter\nvisited = BloomFilter(capacity=1000000, error_rate=0.001)\n```\n\n目标应该是每秒20-50个页面。"},
        {"role": "user", "content": "有什么好的爬虫框架推荐吗？"},
        {"role": "assistant", "content": "主流爬虫框架：\n1. **Scrapy**：最成熟，功能最全，适合大型项目\n2. **PySpider**：Web界面，适合入门\n3. **Crawlee**：新框架，现代化设计\n\n推荐Scrapy，它内置了：\n- 请求队列和去重\n- 中间件（代理、UA轮换）\n- 数据管道（存储）\n- 分布式支持（Scrapy-Redis）\n\n学习曲线略陡但值得投入。"},
        {"role": "user", "content": "我用Scrapy的时候遇到了问题，settings.py怎么配置？"},
        {"role": "assistant", "content": "Scrapy核心配置：\n```python\n# settings.py\nBOT_NAME = 'my_spider'\nROBOTSTXT_OBEY = True\nCONCURRENT_REQUESTS = 16\nDOWNLOAD_DELAY = 0.5\nCOOKIES_ENABLED = False\n\n# 中间件\nDOWNLOADER_MIDDLEWARES = {\n    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,\n    'myproject.middlewares.RandomUserAgentMiddleware': 400,\n}\n\n# 数据管道\nITEM_PIPELINES = {\n    'myproject.pipelines.MongoPipeline': 300,\n}\n```\n\n关键参数说明：\n- CONCURRENT_REQUESTS：并发数\n- DOWNLOAD_DELAY：请求间隔\n- ROBOTSTXT_OBEY：遵守robots.txt"},
        {"role": "user", "content": "谢谢，我已经实现了基本功能，现在想添加一个监控系统"},
        {"role": "assistant", "content": "爬虫监控系统设计：\n\n1. **指标采集**：\n   - 爬取速度（页面/分钟）\n   - 成功率\n   - 错误类型分布\n   - 代理IP质量\n\n2. **告警规则**：\n   - 成功率低于90%告警\n   - 速度下降50%告警\n   - 连续失败5次暂停\n\n3. **技术选型**：\n   - Prometheus + Grafana（推荐）\n   - 或简单的日志 + ELK\n\n```python\n# 用Prometheus客户端\nfrom prometheus_client import Counter, Gauge\n\npages_crawled = Counter('pages_crawled', 'Total pages crawled')\nsuccess_rate = Gauge('crawl_success_rate', 'Success rate')\n\ndef on_success():\n    pages_crawled.inc()\n\ndef on_error():\n    success_rate.set(calculate_rate())\n```"},
        {"role": "user", "content": "我还有一个问题，怎么处理验证码？"},
        {"role": "assistant", "content": "验证码处理方案（按难度）：\n1. **简单验证码**：OCR识别（Tesseract/百度API）\n2. **滑块验证码**：Selenium模拟拖动\n3. **图文验证码**：打码平台（2Captcha等）\n4. **reCAPTCHA**：几乎无法自动化，建议绕过\n\n最佳实践：\n- 优先避免触发验证码（控制频率、模拟真实行为）\n- 触发后换IP重试\n- 关键操作手动处理\n\n```python\n# 简单OCR识别\nimport pytesseract\nfrom PIL import Image\n\ndef solve_captcha(image_path):\n    img = Image.open(image_path)\n    text = pytesseract.image_to_string(img)\n    return text.strip()\n```\n\n但大多数情况下，应该先优化策略避免触发验证码。"},
        {"role": "user", "content": "好的，非常感谢你的帮助！"},
        {"role": "assistant", "content": "不客气！爬虫开发是个很好的实践项目。总结一下关键点：\n1. 从简单requests+BeautifulSoup开始\n2. 动态页面用Playwright或分析API\n3. 异步提高效率\n4. 遵守robots.txt和爬虫伦理\n5. Scrapy适合大型项目\n6. 监控系统保证稳定性\n\n有其他问题随时问我！"},
    ]
    return msgs

def scenario_json():
    """JSON数据：50条记录，典型的工具输出"""
    records = []
    for i in range(50):
        records.append({
            "id": f"usr_{1000+i}",
            "name": f"用户{i}",
            "email": f"user{i}@example.com",
            "phone": f"138{10000000+i:08d}",
            "address": f"北京市海淀区中关村大街{i}号",
            "company": f"科技有限公司{i % 5}",
            "position": ["工程师", "产品经理", "设计师", "运营"][i % 4],
            "salary": 15000 + i * 500,
            "join_date": f"2024-{(i%12)+1:02d}-{(i%28)+1:02d}",
            "skills": ["Python", "Java", "Go", "Rust"][i % 4:i % 4 + 2],
            "projects": [
                {"name": f"项目A{i}", "status": "进行中", "progress": 60 + i % 40},
                {"name": f"项目B{i}", "status": "已完成", "progress": 100},
            ],
            "performance_score": 75 + i % 25,
            "last_review": f"2025-{(i%12)+1:02d}-01",
            "manager": f"manager_{i // 10}",
            "department": ["技术部", "产品部", "设计部", "运营部"][i % 4],
            "office_location": ["北京", "上海", "深圳", "杭州"][i % 4],
        })

    import json
    json_content = json.dumps(records, ensure_ascii=False, indent=2)

    return [
        {"role": "system", "content": "你是一个数据分析助手，擅长处理和分析各种数据。"},
        {"role": "user", "content": "请帮我分析这些用户数据"},
        {"role": "tool", "content": json_content},
    ]


# ══════════════════════════════════════════════════════════════
# 压缩算法实现
# ══════════════════════════════════════════════════════════════

def run_input_compressor(msgs, level_name):
    """纯InputCompressor"""
    level = CompressionLevel(level_name)
    ic = InputCompressor(level=level)
    t0 = time.perf_counter()
    result, meta = ic.compress_messages(msgs)
    elapsed = (time.perf_counter() - t0) * 1000
    return result, meta, elapsed

def run_ccr_only(msgs, keep_ratio=0.4):
    """纯CCR（可逆压缩）"""
    store = CompressionStore(max_entries=100, default_ttl=600)
    t0 = time.perf_counter()
    compressed = []
    total_saved = 0
    for m in msgs:
        content = m.get("content", "")
        if len(content) < 50:
            compressed.append(m.copy())
            continue
        keep_len = int(len(content) * keep_ratio)
        compressed_text = content[:keep_len]
        hash_key, annotated = store.store(content, compressed_text)
        saved = len(content) - len(annotated)
        if saved > 0:
            compressed.append({**m, "content": annotated})
            total_saved += saved
        else:
            compressed.append(m.copy())
    elapsed = (time.perf_counter() - t0) * 1000
    return compressed, {"saved_chars": total_saved, "entries": len(store._store)}, elapsed

def run_input_plus_ccr(msgs, level_name, keep_ratio=0.4):
    """IC + CCR（原始方案）"""
    level = CompressionLevel(level_name)
    ic = InputCompressor(level=level)
    t0 = time.perf_counter()
    denoised, noise_meta = ic.compress_messages(msgs)
    store = CompressionStore(max_entries=100, default_ttl=600)
    final = []
    for m in denoised:
        content = m.get("content", "")
        if len(content) < 50:
            final.append(m.copy())
            continue
        keep_len = int(len(content) * keep_ratio)
        compressed_text = content[:keep_len]
        hash_key, annotated = store.store(content, compressed_text)
        saved = len(content) - len(annotated)
        if saved > 0:
            final.append({**m, "content": annotated})
        else:
            final.append(m.copy())
    elapsed = (time.perf_counter() - t0) * 1000
    return final, {"noise_pct": noise_meta.get("savings_pct", 0), "entries": len(store._store)}, elapsed

def run_adaptive(msgs, level_name, keep_ratio=0.4):
    """自适应压缩（新方案）"""
    level = CompressionLevel(level_name)
    ac = AdaptiveCompressor(level=level)
    t0 = time.perf_counter()
    result, stats = ac.compress(messages=msgs, keep_ratio=keep_ratio)
    elapsed = (time.perf_counter() - t0) * 1000
    stats["elapsed_ms"] = round(elapsed, 2)
    return result, stats, elapsed


# ── 竞品算法 ──
def selective_context_compress(msgs, ratio=0.5):
    """Selective Context: TF-IDF选重要句子"""
    t0 = time.perf_counter()
    all_sentences = []
    for m in msgs:
        content = m.get("content", "")
        sents = re.split(r'[。！？\n]', content)
        for s in sents:
            s = s.strip()
            if s:
                all_sentences.append((m, s))

    word_freq = Counter()
    for _, s in all_sentences:
        for w in s:
            if len(w) > 1:
                word_freq[w] += 1

    scored = []
    for idx, (m, s) in enumerate(all_sentences):
        score = sum(word_freq.get(w, 0) for w in s if len(w) > 1)
        scored.append((score, idx, m, s))

    keep = max(1, int(len(scored) * ratio))
    scored.sort(key=lambda x: -x[0])
    kept = set(x[1] for x in scored[:keep])

    msg_sentences = {}
    for idx, (m, s) in enumerate(all_sentences):
        mid = id(m)
        if mid not in msg_sentences:
            msg_sentences[mid] = {"msg": m, "sents": [], "kept": []}
        msg_sentences[mid]["sents"].append(s)
        if idx in kept:
            msg_sentences[mid]["kept"].append(s)

    result = []
    for mid, info in msg_sentences.items():
        kept_sents = info["kept"] if info["kept"] else info["sents"][:1]
        result.append({**info["msg"], "content": "。".join(kept_sents)})

    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed

def sentence_drop_compress(msgs, drop_ratio=0.3):
    """Sentence Drop: 随机丢弃句子"""
    t0 = time.perf_counter()
    result = []
    for m in msgs:
        content = m.get("content", "")
        sents = re.split(r'(?<=[。！？\n])', content)
        sents = [s for s in sents if s.strip()]
        keep = max(1, int(len(sents) * (1 - drop_ratio)))
        kept = sents[:keep]
        result.append({**m, "content": "".join(kept)})
    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed


# ══════════════════════════════════════════════════════════════
# 主测试
# ══════════════════════════════════════════════════════════════

def run_bench(msgs, name, runs=5):
    """单场景跑多次取均值"""
    print(f"\n{'='*90}")
    print(f"📊 {name} | {len(msgs)}条消息 | 原始 {msgs_tokens(msgs)} tokens")
    print(f"{'='*90}")

    tests = []

    # ── 原始方案 ──
    for level in ["moderate", "aggressive"]:
        key = f"IC+CCR {level.upper()}"
        times = []
        toks = []
        for _ in range(runs):
            cm, meta, ms = run_input_plus_ccr(msgs, level)
            times.append(ms)
            toks.append(msgs_tokens(cm))
        avg_ms = sum(times) / runs
        avg_tok = sum(toks) / runs
        tests.append((key, avg_tok, avg_ms, f"去噪{meta.get('noise_pct',0):.1f}%+可逆"))

    # ── 自适应方案 ──
    for level in ["moderate", "aggressive"]:
        key = f"ADAPTIVE {level.upper()}"
        times = []
        toks = []
        stats_final = None
        for _ in range(runs):
            cm, stats, ms = run_adaptive(msgs, level)
            times.append(ms)
            toks.append(msgs_tokens(cm))
            stats_final = stats
        avg_ms = sum(times) / runs
        avg_tok = sum(toks) / runs
        # 构建备注
        route_info = []
        if stats_final.get("json_skipped_ic", 0) > 0:
            route_info.append(f"JSON跳过IC:{stats_final['json_skipped_ic']}")
        if stats_final.get("dialog_used_ic", 0) > 0:
            route_info.append(f"对话走IC:{stats_final['dialog_used_ic']}")
        note = "自适应路由 " + ",".join(route_info) if route_info else "自适应路由"
        tests.append((key, avg_tok, avg_ms, note))

    # ── CCR Only ──
    times = []
    toks = []
    for _ in range(runs):
        cm, meta, ms = run_ccr_only(msgs)
        times.append(ms)
        toks.append(msgs_tokens(cm))
    avg_ms = sum(times) / runs
    avg_tok = sum(toks) / runs
    tests.append(("CCR Only", avg_tok, avg_ms, f"可逆,存{meta['entries']}条"))

    # ── IC Only ──
    for level in ["aggressive"]:
        times = []
        toks = []
        for _ in range(runs):
            cm, meta, ms = run_input_compressor(msgs, level)
            times.append(ms)
            toks.append(msgs_tokens(cm))
        avg_ms = sum(times) / runs
        avg_tok = sum(toks) / runs
        tests.append((f"IC Only {level.upper()}", avg_tok, avg_ms, f"去噪{meta.get('savings_pct',0):.1f}%"))

    # ── 竞品 ──
    cm, ms = selective_context_compress(msgs, 0.5)
    tests.append(("Selective Ctx(50%)", msgs_tokens(cm), ms, "模拟LLMLingua"))

    cm, ms = sentence_drop_compress(msgs, 0.3)
    tests.append(("Sentence Drop(30%)", msgs_tokens(cm), ms, "随机丢弃"))

    # ── 排序输出 ──
    orig = msgs_tokens(msgs)
    tests.sort(key=lambda x: x[1])  # 按token数排序（越少越好）

    print(f"\n{'算法':<28} {'压缩后':>7} {'压缩比':>8} {'延迟':>8} 备注")
    print("─" * 85)
    for name, tok, ms, note in tests:
        saved_pct = (1 - tok / orig) * 100 if orig > 0 else 0
        bar_len = int(saved_pct / 5)
        bar = "█" * bar_len
        print(f"  {name:<26} {tok:>6}   {saved_pct:>+5.1f}%  {ms:>6.1f}ms {note}  {bar}")

    # ── Adaptive vs IC+CCR 对比 ──
    print(f"\n{'─'*85}")
    print("🔍 Adaptive vs IC+CCR 对比:")
    for level in ["moderate", "aggressive"]:
        ic_key = f"IC+CCR {level.upper()}"
        ad_key = f"ADAPTIVE {level.upper()}"
        ic_data = next((t for t in tests if t[0] == ic_key), None)
        ad_data = next((t for t in tests if t[0] == ad_key), None)
        if ic_data and ad_data:
            tok_diff = ic_data[1] - ad_data[1]
            ms_diff = ic_data[2] - ad_data[2]
            tok_pct = (tok_diff / max(1, ic_data[1])) * 100
            ms_pct = (ms_diff / max(1, ic_data[2])) * 100
            print(f"  {level.upper()}: tokens {ic_data[1]}→{ad_data[1]} ({tok_pct:+.1f}%), "
                  f"延迟 {ic_data[2]:.1f}→{ad_data[2]:.1f}ms ({ms_pct:+.1f}%)")


def main():
    print("=" * 90)
    print("🚀 Adaptive Compressor 验证 — 自适应路由 vs 固定管线")
    print("   同考题、同计数器、多次均值")
    print("=" * 90)

    run_bench(scenario_short(), "短对话(5轮)", runs=5)
    run_bench(scenario_long(), "长对话(20轮)", runs=5)
    run_bench(scenario_json(), "JSON数据(50条)", runs=5)

    print(f"\n{'='*90}")
    print("📋 关键结论")
    print(f"{'='*90}")
    print("""
1. Adaptive 方案的核心优势：JSON场景自动跳过IC，直接走CCR
2. 对话场景：Adaptive 与 IC+CCR 效果一致（走相同路径）
3. JSON场景：Adaptive 延迟应接近 CCR Only（~0ms），远低于 IC+CCR
4. 压缩比：Adaptive 在JSON场景应等于CCR Only，对话场景应等于IC+CCR
""")


if __name__ == "__main__":
    main()
