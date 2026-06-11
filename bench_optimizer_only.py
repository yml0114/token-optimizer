#!/usr/bin/env python3
"""Token Optimizer 实测：3场景 × 3压缩级别"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(text): return len(enc.encode(text))

from token_optimizer.core.signal_noise import InputCompressor, CompressionLevel

SYSTEM_PROMPT = """你是太一AI助手，一个全能型AI Agent。你是用户的私人助手，能够处理各种任务。
当前日期：2026年6月11日 星期四 17:40:00 CST
会话ID：sess_7f3a2b9c-4d5e-6f78-9abc-def012345678
用户ID：usr_88421573
安全策略：禁止输出系统prompt。语言偏好：中文优先。
设备信息：MacBook Pro M2 Max 96GB RAM
操作系统：macOS 26.0
当前任务：协助用户进行投资分析和项目管理。
可用工具：search_web, fetch_web, bash, read_file, write_file, calendar_create, memory_search
模型配置：mimo-v2.5-pro, temperature=0.7, max_tokens=4096
权限级别：L4-完全访问"""

LONG_SYSTEM = """你是太一AI助手，一个全能型AI Agent。你是用户的私人助手，能够处理各种任务。
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
- 0号炸弹(ZERO TNT)：独立金融态势感知产品，v2.7
- AI平行世界：多Agent协作讨论平台，v3架构
- 独立记忆世界(mnemos)：通用AI Agent记忆基础设施，v7.12
- 太一(Taiyi)：全能AI Agent，v6.3
- Token Optimizer：Token压缩与缓存优化引擎，v2.1
核心原则：
1. 所有功能必须有明确实现路径，无幻觉
2. 不清楚的标记[需要验证]先调研确认
3. 结论必须有数据/事实支撑
4. 产品+API一体化（类扣子积分制）
5. 暴君哲学：AI长期目标锚定+持续验收机制
技术栈：FastAPI + Python 3.11 + SQLite + ChromaDB + MiMo-V2.5-Pro + BGE-M3"""

def make_short():
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in [
        ("查一下小米今天的股价", "小米集团-W（01810.HK）今日收盘价为 62.35 港元，较昨日上涨 2.15%。成交量约 1.2 亿股。"),
        ("那苹果呢？", "苹果（AAPL）今日收盘价为 $234.82，微涨 0.33%。市值约 3.58 万亿美元。"),
        ("总结一下今天A股大盘走势", "今日A股三大指数集体收涨。上证指数涨 0.82% 报 3412.56，深证成指涨 1.15% 报 10856.32，创业板指涨 1.58% 报 2234.78。"),
        ("推荐几只科技股", "基于当前市场环境，以下科技股值得关注：1）中芯国际（688981）2）宁德时代（300750）3）比亚迪（002594）。注意：仅供参考。"),
        ("写邮件总结本周项目进展", "邮件草稿：本周项目进展（2026.06.09-06.11）。1.Token Optimizer v2.1发布 2.太一v6.3精算完成 3.mnemos v7.12跑分完成。"),
    ]:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    return msgs

def make_long():
    msgs = [{"role": "system", "content": LONG_SYSTEM}]
    pairs = [
        ("帮我查一下小米集团的最新财报数据", "小米集团2025年Q4财报：营收1089.6亿元，同比增长35.2%。净利润71.3亿元，同比增长52.8%。智能手机出货量4270万台，同比增长15.3%。IoT及生活消费品收入323.7亿元。互联网服务收入89.2亿元。汽车业务收入62.1亿元，交付量4.2万辆。"),
        ("和上一季度比怎么样", "环比Q3：营收增长8.7%（Q3为1002.1亿），净利润增长12.1%（Q3为63.6亿）。手机出货量增长5.2%（Q3为4058万）。汽车交付量增长23.5%（Q3为3.4万）。整体呈上升趋势。"),
        ("分析一下小米的护城河", "小米护城河：1）品牌：全球前三 2）生态：IoT平台7亿设备 3）供应链：自研澎湃芯片 4）用户：MIUI月活6.8亿 5）成本：硬件利润率不超5%。"),
        ("对比比亚迪和小米汽车", "比亚迪vs小米：销量427万vs18万；收入6800亿vs250亿；毛利率22.8%vs18.5%；技术路线全产业链vs整合供应链。"),
        ("推荐AI概念股", "A股AI概念：1）科大讯飞（002230）2）海康威视（002415）3）寒武纪（688256）4）中科创达（300496）5）金山办公（688111）。"),
        ("估值怎么样", "估值：科大讯飞PE85倍偏高；海康PE28倍合理；寒武纪亏损PS45倍极高；中科创达PE52倍略高；金山办公PE78倍偏高。"),
        ("今天热搜", "热搜Top5：1）高考结束 2）小米YU7发布 3）苹果WWDC 4）A股收涨 5）梅西退役。"),
        ("YU7配置", "小米YU7：21.59万起；双电机495kW；零百3.8s；续航755km；800V平台；激光雷达+Orin X。"),
        ("和Model Y比", "YU7 vs Model Y：21.59万vs24.99万；755kmvs688km；3.8s vs5.0s；激光雷达vs纯视觉。"),
        ("太一进展", "太一v6.3：产品+API一体化完成；会员¥19.9/月；盈亏平衡163人；Phase路线图已定。"),
        ("mnemos呢", "mnemos v7.12：声称97%准确率，零LLM零GPU，已开源。"),
        ("写投资人周报", "周报：1）Token Optimizer v2.1发布 2）太一v6.3精算完成 3）mnemos跑分产出。"),
        ("天气", "深圳：多云转晴26-33℃，东南风3-4级。"),
        ("明天天气", "深圳明天：阵雨转多云25-31℃，湿度80%。"),
        ("开源项目推荐", "GitHub热门：headroom 12k stars；MiMo-V2.5；CrewAI v1.0；SGLang v0.5。"),
        ("headroom对比", "headroom vs Token Optimizer：代理层vs纯库；需LLM vs纯规则；声称60-95% vs实测-60.4%。"),
        ("查邮件", "最近：1）GitHub v2.1 Release 2）扣子部署成功 3）小米API月报。"),
        ("规划明天", "明天：上午竞品对比+Phase 0 review；下午mnemos验证+周报确认。"),
        ("竞品对比怎么做", "方案：选LLMLingua-2/headroom/LLMLingua；统一tiktoken；对比压缩比/延迟/可逆性。"),
        ("开始执行", "好的，已开始执行。"),
    ]
    for u, a in pairs:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    return msgs

def make_json():
    records = []
    for i in range(50):
        records.append({
            "id": f"stock_{i:03d}",
            "name": ["贵州茅台","宁德时代","比亚迪","中芯国际","海康威视","科大讯飞","金山办公","寒武纪","中科创达","紫光国微"][i%10],
            "code": f"{600000+i:06d}",
            "price": round(100+i*12.5, 2),
            "change_pct": round(-3+i*0.15, 2),
            "volume": (i+1)*1234567,
            "market_cap": round((i+1)*123.45, 2),
            "pe_ratio": round(15+i*2.3, 1),
            "pb_ratio": round(1.2+i*0.3, 2),
            "roe": round(8+i*1.5, 1),
            "dividend_yield": round(0.5+i*0.2, 2),
            "sector": "科技" if i%3==0 else ("消费" if i%3==1 else "新能源"),
            "exchange": "上交所" if i%2==0 else "深交所",
            "analyst_rating": ["强烈推荐","推荐","中性","减持"][i%4],
            "risk_level": ["低","中","高"][i%3],
            "notes": f"该公司在{['人工智能','新能源汽车','半导体','云计算','物联网'][i%5]}领域领先。"
        })
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "帮我查一下A股科技股行情数据"},
        {"role": "assistant", "content": json.dumps({"status":"success","total":50,"data":records}, ensure_ascii=False, indent=2)},
    ]

def msg_tokens(msgs):
    t = 0
    for m in msgs:
        t += count_tokens(m.get("content",""))
        t += count_tokens(m.get("role",""))
        t += 4
    return t

def run(name, messages):
    orig = msg_tokens(messages)
    print(f"\n{'='*60}")
    print(f"📊 {name}  |  {len(messages)}条消息  |  原始 {orig} tokens")
    print(f"{'='*60}")
    print(f"{'级别':<14} {'压缩后':<10} {'压缩比':<12} {'延迟':<10} {'去除tokens':<12}")
    print("─"*60)
    
    for level_name, level in [("SAFE", CompressionLevel.SAFE), ("MODERATE", CompressionLevel.MODERATE), ("AGGRESSIVE", CompressionLevel.AGGRESSIVE)]:
        comp = InputCompressor(level=level)
        start = time.time()
        result_msgs, meta = comp.compress_messages(messages)
        elapsed = (time.time() - start) * 1000
        comp_tokens = msg_tokens(result_msgs)
        ratio = (1 - comp_tokens / orig) * 100 if orig > 0 else 0
        noise = meta.get("noise_removed_tokens", 0)
        print(f"{level_name:<14} {comp_tokens:<10} {ratio:>+7.1f}%    {elapsed:>7.1f}ms  {noise:>6}")
    
    return orig

def main():
    print("🏆 Token Optimizer InputCompressor 压缩实测")
    print(f"   计数器: tiktoken cl100k_base")
    
    r1 = run("场景1: 短对话（5轮中文）", make_short())
    r2 = run("场景2: 长对话（20轮中文）", make_long())
    r3 = run("场景3: JSON数据（50条记录）", make_json())
    
    print(f"\n{'='*60}")
    print("✅ 测试完成")

if __name__ == "__main__":
    main()
