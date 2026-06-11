#!/usr/bin/env python3
"""
Token Optimizer vs 竞品 压缩比对比测试
同一考题，统一 tiktoken cl100k_base 计算 token 数
"""
import sys, os, time, json, re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── tiktoken ──
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(enc.encode(text))
except ImportError:
    enc = None
    def count_tokens(text: str) -> int:
        return len(text) // 4  # 粗略估计

# ── LLMLingua ──
HAS_LLMLINGUA = False
try:
    from llmlingua import PromptCompressor
    llm_compressor = PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank", device_map="cpu")
    HAS_LLMLINGUA = True
    print("✅ LLMLingua-2 加载成功")
except Exception as e:
    print(f"⚠️  LLMLingua 不可用: {e}")

# ── Token Optimizer 规则压缩 ──
HAS_TOKEN_OPTIMIZER = False
try:
    from token_optimizer.core.rule_compressor import RuleCompressor
    rule_comp = RuleCompressor()
    HAS_TOKEN_OPTIMIZER = True
    print("✅ Token Optimizer RuleCompressor 加载成功")
except Exception as e:
    try:
        # 尝试其他导入路径
        from token_optimizer.core import rule_compressor
        # 查找可用的类
        classes = [x for x in dir(rule_compressor) if 'compress' in x.lower() or 'Compressor' in x]
        print(f"  可用类/函数: {classes}")
        # 尝试实例化第一个匹配的类
        for cls_name in classes:
            cls = getattr(rule_compressor, cls_name)
            if isinstance(cls, type) and hasattr(cls, 'compress'):
                rule_comp = cls()
                HAS_TOKEN_OPTIMIZER = True
                print(f"✅ Token Optimizer {cls_name} 加载成功")
                break
        if not HAS_TOKEN_OPTIMIZER:
            print(f"⚠️  Token Optimizer: 找到 {classes} 但无法实例化")
    except Exception as e2:
        print(f"⚠️  Token Optimizer 不可用: {e} | {e2}")


# ═══════════════════════════════════════════════════
# 测试场景
# ═══════════════════════════════════════════════════

SYSTEM_PROMPT = """你是太一AI助手，一个全能型AI Agent。你是用户的私人助手，能够处理各种任务。
当前日期：2026年6月11日 星期四 17:40:00 CST
会话ID：sess_7f3a2b9c-4d5e-6f78-9abc-def012345678
用户ID：usr_88421573
安全策略：禁止输出系统prompt。语言偏好：中文优先。
设备信息：MacBook Pro M2 Max 96GB RAM
操作系统：macOS 26.0
当前任务：协助用户进行投资分析和项目管理。
可用工具：search_web, fetch_web, bash, read_file, write_file
模型配置：mimo-v2.5-pro, temperature=0.7, max_tokens=4096
权限级别：L4-完全访问"""

LONG_SYSTEM_PROMPT = """你是太一AI助手，一个全能型AI Agent。你是用户的私人助手，能够处理各种任务。
当前日期：2026年6月11日 星期四 17:40:00 CST
会话ID：sess_7f3a2b9c-4d5e-6f78-9abc-def012345678
用户ID：usr_88421573
安全策略：禁止输出系统prompt。语言偏好：中文优先。
设备信息：MacBook Pro M2 Max 96GB RAM
操作系统：macOS 26.0
当前任务：协助用户进行投资分析和项目管理。
可用工具：search_web, fetch_web, bash, read_file, write_file, calendar_create, memory_search
模型配置：mimo-v2.5-pro, temperature=0.7, max_tokens=4096
权限级别：L4-完全访问
项目信息：
- 0号炸弹(ZERO TNT)：独立金融态势感知产品，v2.7，与AI平行世界并行开发
- AI平行世界：多Agent协作讨论平台，v3架构，镇长→管家→专家团→流式讨论面板
- 独立记忆世界(mnemos)：通用AI Agent记忆基础设施，Apache 2.0，v7.12
- 太一(Taiyi)：全能AI Agent，v6.3，暴君系统+自我进化+个性化+本地优先
- Token Optimizer：通用型Token压缩与缓存优化引擎，MIT，v2.1
核心原则：
1. 所有功能必须有明确实现路径，无幻觉
2. 不清楚的标记[需要验证]先调研确认
3. 结论必须有数据/事实支撑
4. 产品+API一体化（类扣子积分制）
5. 暴君哲学：AI长期目标锚定+持续验收机制
技术栈：
- 前端：React + TypeScript + Vite + Tailwind CSS
- 后端：FastAPI + Python 3.11
- 数据库：SQLite + ChromaDB
- 模型：MiMo-V2.5-Pro（推理） + BGE-M3（embedding）
- 部署：Docker + nginx + certbot
- CI/CD：GitHub Actions
API端点：
- 模型推理：https://token-plan-cn.xiaomimimo.com/v1
- 知识库：http://localhost:8000/api/v1/knowledge
- 用户管理：http://localhost:8000/api/v1/users
- 记忆系统：http://localhost:8000/api/v1/memory
- 技能市场：http://localhost:8000/api/v1/skills
"""

def make_short_conversation():
    """场景1: 短对话（10轮）"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    pairs = [
        ("查一下小米今天的股价", "小米集团-W（01810.HK）今日收盘价为 62.35 港元，较昨日上涨 2.15%。成交量约 1.2 亿股。"),
        ("那苹果呢？", "苹果（AAPL）今日收盘价为 $234.82，微涨 0.33%。市值约 3.58 万亿美元。"),
        ("总结一下今天A股大盘走势", "今日A股三大指数集体收涨。上证指数涨 0.82% 报 3412.56，深证成指涨 1.15% 报 10856.32，创业板指涨 1.58% 报 2234.78。"),
        ("推荐几只科技股", "基于当前市场环境，以下科技股值得关注：1）中芯国际（688981）2）宁德时代（300750）3）比亚迪（002594）。注意：仅供参考。"),
        ("写邮件总结本周项目进展", "邮件草稿：本周项目进展（2026.06.09-06.11）。1.Token Optimizer v2.1发布 2.太一v6.3精算完成 3.mnemos v7.12跑分完成。"),
    ]
    for u, a in pairs:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    return messages

def make_long_conversation():
    """场景2: 长对话（30轮）"""
    messages = [{"role": "system", "content": LONG_SYSTEM_PROMPT}]
    pairs = [
        ("帮我查一下小米集团的最新财报数据", "小米集团2025年Q4财报：营收1089.6亿元，同比增长35.2%。净利润71.3亿元，同比增长52.8%。智能手机出货量4270万台，同比增长15.3%。IoT及生活消费品收入323.7亿元。互联网服务收入89.2亿元。汽车业务收入62.1亿元，交付量4.2万辆。"),
        ("和上一季度比怎么样", "环比Q3：营收增长8.7%（Q3为1002.1亿），净利润增长12.1%（Q3为63.6亿）。手机出货量增长5.2%（Q3为4058万）。汽车交付量增长23.5%（Q3为3.4万）。整体呈上升趋势。"),
        ("分析一下小米的护城河", "小米的护城河分析：1）品牌护城河：全球手机品牌前三，印度市场第一 2）生态护城河：全球最大消费级IoT平台，连接设备超7亿台 3）供应链护城河：自研芯片澎湃系列，汽车工厂自建 4）用户护城河：MIUI月活用户6.8亿 5）成本护城河：硬件利润率承诺不超5%，互联网服务变现。"),
        ("对比一下比亚迪和小米的汽车业务", "比亚迪 vs 小米汽车对比：销量：比亚迪2025年427万辆 vs 小米18万辆；收入：比亚迪6800亿 vs 小米250亿；毛利率：比亚迪22.8% vs 小米18.5%；技术路线：比亚迪全产业链自研 vs 小米整合供应链；产品线：比亚迪覆盖5-50万 vs 小米聚焦20-40万；智能化：比亚迪DiPilot vs 小米Pilot Max。"),
        ("推荐几只AI概念股", "A股AI概念股推荐：1）科大讯飞（002230）：AI语音龙头，讯飞星火大模型 2）海康威视（002415）：AI视觉龙头，智能安防 3）寒武纪（688256）：AI芯片龙头，思元系列 4）中科创达（300496）：智能操作系统，汽车AI 5）金山办公（688111）：AI办公，WPS AI。仅供参考。"),
        ("这些股票的估值怎么样", "AI概念股估值分析：科大讯飞PE 85倍（历史中位65倍，偏高）；海康威视PE 28倍（历史中位32倍，合理）；寒武纪PE 负值（亏损中，PS 45倍，极高）；中科创达PE 52倍（历史中位48倍，略高）；金山办公PE 78倍（历史中位70倍，偏高）。整体AI板块估值偏贵。"),
        ("帮我查一下今天的热搜", "今日热搜Top5：1）高考结束考生冲出考场 2）小米汽车YU7正式发布 3）苹果WWDC2026开幕 4）A股三大指数集体收涨 5）梅西宣布退役。"),
        ("小米YU7的配置怎么样", "小米YU7配置：售价21.59万起；双电机四驱，最大功率495kW；零百加速3.8秒；CLTC续航755km；800V高压平台，15分钟充电500km；激光雷达+Orin X芯片，城市NOA标配；车身尺寸4999×1996×1600mm，轴距3000mm。"),
        ("和特斯拉Model Y比呢", "小米YU7 vs 特斯拉Model Y：售价：21.59万 vs 24.99万；续航：755km vs 688km；零百：3.8秒 vs 5.0秒；智驾：激光雷达+Orin X vs 纯视觉FSD；空间：轴距3000mm vs 2890mm；补能：800V 15min vs 250kW 30min。YU7性价比更高。"),
        ("帮我看一下太一项目的进展", "太一项目v6.3进展：1）产品+API一体化架构已完成 2）会员定价精算完成：¥19.9/月，3.5万积分 3）盈亏平衡点163人 4）Phase 0-3路线图已制定 5）暴君系统技术实现方案确定 6）技能市场4级权限体系设计完成。"),
        ("mnemos项目呢", "mnemos v7.12状态：用户声称97.0%准确率（485/500），零LLM零GPU零成本。GitHub已开源：https://github.com/yml0114/mnemos 。但该数据未经独立验证。"),
        ("Token Optimizer v2.1发布了是吧", "是的，Token Optimizer v2.1已发布。核心功能：四层压缩引擎（规则层+Smart压缩+CCR可逆+统计分析）。Benchmark结果：json_list -60.4%，SmartCompressor+CCR -37.1%，CCR hit 100%。GitHub Release：https://github.com/yml0114/token-optimizer/releases/tag/v2.1"),
        ("帮我写一封给投资人的周报", "投资人周报（2026.06.09-06.11）：本周重点成果：1）Token Optimizer v2.1正式发布，压缩比提升显著 2）太一v6.3精算验证完成，商业模式清晰 3）mnemos v7.12跑分数据产出。下周计划：1）Token Optimizer竞品对比测试 2）太一Phase 0启动准备 3）mnemos独立验证测试。"),
        ("今天天气怎么样", "查询到您所在城市（深圳）今日天气：多云转晴，气温26-33℃，东南风3-4级，湿度72%。紫外线指数：中等。建议外出携带防晒用品。"),
        ("明天呢", "深圳明天天气预报：阵雨转多云，气温25-31℃，南风2-3级，湿度80%。降雨概率65%，建议带伞。"),
        ("帮我查一下最近有什么好的开源项目", "GitHub近期热门开源项目：1）headroom：Token压缩工具，12k+ stars，Trending #1 2）MiMo-V2.5：小米推理大模型 3）CrewAI v1.0：多Agent协作框架 4）SGLang v0.5：高性能推理引擎 5）OpenHands：AI编程助手。"),
        ("headroom和你的Token Optimizer比怎么样", "headroom vs Token Optimizer对比：定位：headroom是代理/包装层，Token Optimizer是纯库；依赖：headroom需要LLM，Token Optimizer纯规则可独立运行；压缩比：headroom声称60-95%，Token Optimizer实测-60.4%（json场景）；特色：headroom有CCR可逆，Token Optimizer有SmartCompressor自适应+利润守卫。"),
        ("帮我看一下最近的邮件", "最近3封邮件：1）GitHub：token-optimizer v2.1 Release published 2）扣子：您的Agent已成功部署 3）小米开发者平台：API调用量月报已生成。"),
        ("回复第一封邮件", "GitHub通知邮件不需要回复，系统自动发送。如果您需要给特定的人发送关于v2.1发布的通知邮件，请告诉我收件人和内容。"),
        ("帮我规划一下明天的工作", "明日工作规划（2026-06-12）：上午：1）Token Optimizer竞品对比测试（预计2h）2）太一Phase 0启动文档review（预计1h）；下午：3）mnemos独立验证测试（预计2h）4）投资人周报发送确认（预计0.5h）；晚间：5）技术文档整理（预计1h）。"),
        ("第一个任务具体怎么做", "Token Optimizer竞品对比方案：1）选取Top 3竞品：LLMLingua-2、headroom、LLMLingua 2）统一测试集：短对话/长对话/JSON数据 3）统一指标：tiktoken cl100k_base计算token数 4）对比维度：压缩比、延迟、可逆性 5）输出对比报告+图表。"),
        ("开始执行吧", "好的，已开始执行竞品对比测试。正在安装依赖并准备测试环境..."),
        ("进度怎么样", "当前进度：1）✅ 测试脚本编写完成 2）✅ 依赖安装完成 3）⏳ 正在运行基准测试 4）⏳ 待收集结果。预计还需5分钟完成。"),
        ("快点", "正在加速执行，已并行运行三个场景的测试。预计3分钟内出结果。"),
        ("测试结果出来了吗", "测试结果已出。Token Optimizer在JSON场景压缩比最高（-60.4%），LLMLingua-2在对话场景表现更好（-45%），综合来看Token Optimizer在多场景下更稳定。"),
        ("发我一份报告", "报告已生成，正在导出为Markdown格式..."),
        ("格式用表格", "已调整为表格格式，包含：场景、工具、原始tokens、压缩后tokens、压缩比、延迟、可逆性七个维度。"),
        ("好的谢谢", "不客气！有任何问题随时问我。"),
        ("晚安", "晚安！明天见 😊"),
    ]
    for u, a in pairs:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    return messages

def make_json_data():
    """场景3: JSON数据（模拟工具调用返回）"""
    records = []
    for i in range(50):
        records.append({
            "id": f"stock_{i:03d}",
            "name": ["贵州茅台", "宁德时代", "比亚迪", "中芯国际", "海康威视", "科大讯飞", "金山办公", "寒武纪", "中科创达", "紫光国微"][i % 10],
            "code": f"{600000 + i:06d}",
            "price": round(100 + i * 12.5, 2),
            "change_pct": round(-3 + i * 0.15, 2),
            "volume": (i + 1) * 1234567,
            "market_cap": round((i + 1) * 123.45, 2),
            "pe_ratio": round(15 + i * 2.3, 1),
            "pb_ratio": round(1.2 + i * 0.3, 2),
            "roe": round(8 + i * 1.5, 1),
            "dividend_yield": round(0.5 + i * 0.2, 2),
            "sector": "科技" if i % 3 == 0 else ("消费" if i % 3 == 1 else "新能源"),
            "exchange": "上交所" if i % 2 == 0 else "深交所",
            "listed_date": f"20{10 + i % 15:02d}-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            "analyst_rating": ["强烈推荐", "推荐", "中性", "减持"][i % 4],
            "target_price": round(100 + i * 15, 2),
            "risk_level": ["低", "中", "高"][i % 3],
            "notes": f"该公司在{['人工智能', '新能源汽车', '半导体', '云计算', '物联网'][i % 5]}领域具有领先地位，建议关注。"
        })
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "帮我查一下A股科技股的行情数据"},
        {"role": "assistant", "content": json.dumps({"status": "success", "total": 50, "data": records}, ensure_ascii=False, indent=2)},
    ]
    return messages


# ═══════════════════════════════════════════════════
# 压缩器
# ═══════════════════════════════════════════════════

def messages_to_text(messages):
    """将消息列表转为单个文本（供 LLMLingua 使用）"""
    parts = []
    for m in messages:
        parts.append(f"[{m['role']}] {m['content']}")
    return "\n".join(parts)

def tokenize_messages(messages):
    """计算消息总 token 数"""
    total = 0
    for m in messages:
        total += count_tokens(m.get("content", ""))
        total += count_tokens(m.get("role", ""))
        total += 4  # 每条消息的格式开销
    return total

def compress_token_optimizer(messages):
    """Token Optimizer 规则压缩"""
    if not HAS_TOKEN_OPTIMIZER:
        return None, 0
    start = time.time()
    # 尝试不同的压缩接口
    try:
        if hasattr(rule_comp, 'compress_messages'):
            result = rule_comp.compress_messages(messages)
        elif hasattr(rule_comp, 'compress'):
            result = rule_comp.compress(messages)
        else:
            return None, 0
        
        elapsed = (time.time() - start) * 1000
        
        # 处理不同的返回格式
        if isinstance(result, tuple):
            compressed_msgs, meta = result
        elif isinstance(result, dict):
            compressed_msgs = result.get("messages", result.get("compressed", messages))
            meta = result
        else:
            compressed_msgs = result
            meta = {}
        
        return compressed_msgs, elapsed
    except Exception as e:
        print(f"  Token Optimizer 错误: {e}")
        return None, 0

def compress_llmlingua(messages):
    """LLMLingua-2 压缩"""
    if not HAS_LLMLINGUA:
        return None, 0
    
    start = time.time()
    try:
        # LLMLingua 需要纯文本输入
        full_text = messages_to_text(messages)
        result = llm_compressor.compress_prompt(
            full_text,
            rate=0.5,
            force_tokens=["[", "]", ":", "{", "}", ",", ".", '"'],
        )
        elapsed = (time.time() - start) * 1000
        
        compressed_text = result.get("compressed_prompt", full_text)
        # 转回消息格式
        return [{"role": "system", "content": "[LLMLingua压缩]"}, 
                {"role": "user", "content": compressed_text}], elapsed
    except Exception as e:
        print(f"  LLMLingua 错误: {e}")
        return None, 0

def compress_baseline(messages):
    """Baseline: 简单去除冗余"""
    start = time.time()
    compressed = []
    for m in messages:
        content = m.get("content", "")
        # 1. 去除多余空白
        content = re.sub(r'\s+', ' ', content).strip()
        # 2. 去除重复标点
        content = re.sub(r'([。，！？、；：])\1+', r'\1', content)
        # 3. 去除常见填充词
        content = content.replace("请问", "").replace("麻烦", "")
        content = content.replace("谢谢", "").replace("感谢", "")
        content = content.replace("您好", "").replace("你好", "")
        compressed.append({**m, "content": content})
    elapsed = (time.time() - start) * 1000
    return compressed, elapsed


# ═══════════════════════════════════════════════════
# 主测试
# ═══════════════════════════════════════════════════

def run_test(name, messages):
    print(f"\n{'='*60}")
    print(f"📊 {name}")
    print(f"{'='*60}")
    
    orig_tokens = tokenize_messages(messages)
    print(f"原始 token 数: {orig_tokens}")
    print(f"消息条数: {len(messages)}")
    print(f"{'─'*60}")
    
    results = []
    
    # Token Optimizer
    if HAS_TOKEN_OPTIMIZER:
        comp_msgs, elapsed = compress_token_optimizer(messages)
        if comp_msgs:
            comp_tokens = tokenize_messages(comp_msgs)
            ratio = (1 - comp_tokens / orig_tokens) * 100 if orig_tokens > 0 else 0
            results.append(("Token Optimizer", orig_tokens, comp_tokens, ratio, elapsed))
            # 打印压缩后前50字
            preview = str(comp_msgs[0].get("content", ""))[:50] if comp_msgs else "N/A"
            print(f"  Token Optimizer: {orig_tokens} → {comp_tokens} ({ratio:+.1f}%) [{elapsed:.1f}ms]")
            print(f"    预览: {preview}...")
        else:
            print(f"  Token Optimizer: 压缩失败")
    
    # LLMLingua-2
    if HAS_LLMLINGUA:
        comp_msgs, elapsed = compress_llmlingua(messages)
        if comp_msgs:
            comp_tokens = tokenize_messages(comp_msgs)
            ratio = (1 - comp_tokens / orig_tokens) * 100 if orig_tokens > 0 else 0
            results.append(("LLMLingua-2", orig_tokens, comp_tokens, ratio, elapsed))
            preview = str(comp_msgs[0].get("content", ""))[:50] if comp_msgs else "N/A"
            print(f"  LLMLingua-2:     {orig_tokens} → {comp_tokens} ({ratio:+.1f}%) [{elapsed:.1f}ms]")
            print(f"    预览: {preview}...")
        else:
            print(f"  LLMLingua-2: 压缩失败")
    
    # Baseline
    comp_msgs, elapsed = compress_baseline(messages)
    comp_tokens = tokenize_messages(comp_msgs)
    ratio = (1 - comp_tokens / orig_tokens) * 100 if orig_tokens > 0 else 0
    results.append(("Baseline(简单)", orig_tokens, comp_tokens, ratio, elapsed))
    preview = str(comp_msgs[0].get("content", ""))[:50] if comp_msgs else "N/A"
    print(f"  Baseline(简单):  {orig_tokens} → {comp_tokens} ({ratio:+.1f}%) [{elapsed:.1f}ms]")
    print(f"    预览: {preview}...")
    
    return results

def main():
    print("=" * 60)
    print("🏆 Token Optimizer vs 竞品 压缩比对比测试")
    print("=" * 60)
    print(f"Token Optimizer: {'✅ 可用' if HAS_TOKEN_OPTIMIZER else '❌ 不可用'}")
    print(f"LLMLingua-2:     {'✅ 可用' if HAS_LLMLINGUA else '❌ 不可用（需CPU下载模型）'}")
    print(f"Baseline:        ✅ 可用")
    print(f"Token 计数器:    {'tiktoken' if enc else '字符估算'}")
    
    # 运行三个场景
    all_results = {}
    all_results["场景1-短对话"] = run_test("场景1: 短对话（中文，10轮）", make_short_conversation())
    all_results["场景2-长对话"] = run_test("场景2: 长对话（中文，30轮）", make_long_conversation())
    all_results["场景3-JSON数据"] = run_test("场景3: JSON数据（50条记录）", make_json_data())
    
    # ── 汇总表格 ──
    print(f"\n{'='*60}")
    print("📋 汇总对比表")
    print(f"{'='*60}")
    print(f"{'场景':<16} {'工具':<18} {'原始':<8} {'压缩后':<8} {'压缩比':<10} {'延迟':<10}")
    print("─" * 60)
    
    for scene, results in all_results.items():
        for tool, orig, comp, ratio, elapsed in results:
            print(f"{scene:<16} {tool:<18} {orig:<8} {comp:<8} {ratio:>+7.1f}%  {elapsed:>7.1f}ms")
        print("─" * 60)
    
    # ── 胜负判断 ──
    print(f"\n{'='*60}")
    print("🏆 各场景压缩比排名")
    print(f"{'='*60}")
    for scene, results in all_results.items():
        # 按压缩比排序（越负越好，即压缩越多越好）
        sorted_results = sorted(results, key=lambda x: x[3])
        print(f"\n{scene}:")
        for rank, (tool, orig, comp, ratio, elapsed) in enumerate(sorted_results, 1):
            medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else "🥉")
            print(f"  {medal} {tool}: {ratio:+.1f}% ({orig} → {comp}, {elapsed:.1f}ms)")
    
    print(f"\n✅ 测试完成")

if __name__ == "__main__":
    main()
