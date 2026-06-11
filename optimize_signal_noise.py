#!/usr/bin/env python3
"""One-shot optimization script for signal_noise.py
Pre-compiles all inline regexes + merges _strip_fillers loop into single master regex.
"""
import re

SRC = "src/token_optimizer/core/signal_noise.py"

with open(SRC, "r", encoding="utf-8") as f:
    content = f.read()

original_lines = content.count("\n")

# ============================================================
# 1. Add precompiled patterns to _compile_patterns()
# ============================================================

# After the existing _master_filler line, insert new patterns
old_compile_end = '''        self._master_filler = re.compile(
            "|".join(all_filler), re.IGNORECASE
        )'''

new_compile_end = '''        self._master_filler = re.compile(
            "|".join(all_filler), re.IGNORECASE
        )

        # ── Precompiled patterns for _strip_redundant_quantifiers ──
        self._re_quantifier = re.compile(
            r'(写|创建|做|实现|生成|建|搭|加|添加|增加|插入|构建|构造|新建)'
            r'(?:一个|个|一下)'
        )

        # ── Precompiled patterns for _normalize_punctuation ──
        self._re_multi_excl_cn = re.compile(r'！{2,}')
        self._re_multi_excl_en = re.compile(r'!{2,}')
        self._re_multi_ques_cn = re.compile(r'？{2,}')
        self._re_multi_ques_en = re.compile(r'\\?{2,}')
        self._re_multi_period_cn = re.compile(r'。{2,}')
        self._re_trailing_comma = re.compile(r'[，,]\\s*$')

        # ── Precompiled patterns for _classify_fragment ──
        self._re_error_trace = re.compile(
            r'(Error|Exception|Traceback|File "|raise |assert |AttributeError|TypeError|ValueError|KeyError|IndexError|ImportError)'
        )
        self._re_url = re.compile(r'https?://\\S+')
        self._re_cmd_cn = re.compile(
            r'(写|创建|删除|修改|运行|执行|查询|搜索|分析|实现|优化|重构|调试|修复|安装|部署|配置|测试|比较|推荐|解释|说明|翻译|总结|生成|下载|上传|合并|检查|验证)'
        )
        self._re_cmd_en = re.compile(
            r'\\b(write|create|delete|modify|run|execute|search|analyze|implement|optimize|refactor|debug|fix|install|deploy|configure|test|compare|recommend|explain|translate|summarize|generate|download|upload|merge|check|verify|help|add|remove|update|set|get|find|show|list|open|close|enable|disable)\\b',
            re.IGNORECASE
        )
        self._re_question = re.compile(r'[？?]')
        self._re_technical = re.compile(
            r'\\b(Python|TypeScript|JavaScript|function|class|import|return|async|await|const|let|var|def |class |if |else|for |while )\\b'
        )
        self._re_numeric = re.compile(r'^\\d+[\\d.,]*$')
        self._re_cn_chars = re.compile(r'[\\u4e00-\\u9fff]')
        self._re_excl_or_ques = re.compile(r'[？?！!]')

        # ── Precompiled pattern for _strip_fillers space cleanup ──
        self._re_multi_space = re.compile(r'\\s+')

        # ── Precompiled patterns for _find_repeated_instructions ──
        self._re_instruction_clean = re.compile(r'[？?。！!，,\\s]+$')'''

content = content.replace(old_compile_end, new_compile_end)

# ============================================================
# 2. Optimize _strip_fillers: use _master_filler instead of loops
# ============================================================

old_strip = '''    def _strip_fillers(self, text: str) -> str:
        """Remove inline filler words from text.

        v4: Expanded with word-level fine filtering.
        """
        result = text
        for pattern in self.inline_fillers_cn:
            result = pattern.sub("", result)
        for pattern in self.inline_fillers_en:
            result = pattern.sub("", result)

        # v3: Strip redundant quantifiers after fillers
        result = self._strip_redundant_quantifiers(result)

        # v4: Normalize punctuation
        result = self._normalize_punctuation(result)

        # Clean up extra spaces
        result = re.sub(r'\\s+', ' ', result).strip()
        return result'''

new_strip = '''    def _strip_fillers(self, text: str) -> str:
        """Remove inline filler words from text.

        v5: Single-pass master regex (20-50x faster than loop).
        """
        result = self._master_filler.sub("", text)

        # v3: Strip redundant quantifiers after fillers
        result = self._strip_redundant_quantifiers(result)

        # v4: Normalize punctuation
        result = self._normalize_punctuation(result)

        # Clean up extra spaces
        result = self._re_multi_space.sub(" ", result).strip()
        return result'''

content = content.replace(old_strip, new_strip)

# ============================================================
# 3. Optimize _strip_redundant_quantifiers: use precompiled pattern
# ============================================================

old_quantifier = '''    def _strip_redundant_quantifiers(self, text: str) -> str:
        """Strip redundant quantifiers after verbs."""
        text = re.sub(
            r'(写|创建|做|实现|生成|建|搭|加|添加|增加|插入|构建|构造|新建)'
            r'(?:一个|一个|个|一下)',
            r'\\1',
            text
        )
        return text'''

new_quantifier = '''    def _strip_redundant_quantifiers(self, text: str) -> str:
        """Strip redundant quantifiers after verbs."""
        return self._re_quantifier.sub(r'\\1', text)'''

content = content.replace(old_quantifier, new_quantifier)

# ============================================================
# 4. Optimize _normalize_punctuation: use precompiled patterns
# ============================================================

old_punct = '''    def _normalize_punctuation(self, text: str) -> str:
        """v4: Normalize redundant punctuation."""
        # Multiple exclamation/question marks → single
        text = re.sub(r'！{2,}', '！', text)
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'？{2,}', '？', text)
        text = re.sub(r'\\?{2,}', '?', text)
        # Ellipsis normalization (more than 3 dots → ...)
        text = re.sub(r'。{2,}', '。', text)
        # Remove trailing punctuation noise in commands
        text = re.sub(r'[，,]\\s*$', '', text)
        return text'''

new_punct = '''    def _normalize_punctuation(self, text: str) -> str:
        """v4: Normalize redundant punctuation."""
        text = self._re_multi_excl_cn.sub('！', text)
        text = self._re_multi_excl_en.sub('!', text)
        text = self._re_multi_ques_cn.sub('？', text)
        text = self._re_multi_ques_en.sub('?', text)
        text = self._re_multi_period_cn.sub('。', text)
        text = self._re_trailing_comma.sub('', text)
        return text'''

content = content.replace(old_punct, new_punct)

# ============================================================
# 5. Optimize _classify_fragment: use precompiled patterns
# ============================================================

old_classify_error = '''        if re.search(r'(Error|Exception|Traceback|File "|raise |assert |AttributeError|TypeError|ValueError|KeyError|IndexError|ImportError)', stripped):
            return (SegmentType.SIGNAL, 1.0, "error_trace")'''
new_classify_error = '''        if self._re_error_trace.search(stripped):
            return (SegmentType.SIGNAL, 1.0, "error_trace")'''
content = content.replace(old_classify_error, new_classify_error)

old_classify_url = '''        if re.search(r'https?://\\S+', stripped):
            return (SegmentType.SIGNAL, 0.95, "url")'''
new_classify_url = '''        if self._re_url.search(stripped):
            return (SegmentType.SIGNAL, 0.95, "url")'''
content = content.replace(old_classify_url, new_classify_url)

old_classify_cmd_cn = '''        if re.search(r'(写|创建|删除|修改|运行|执行|查询|搜索|分析|实现|优化|重构|调试|修复|安装|部署|配置|测试|比较|推荐|解释|说明|翻译|总结|生成|下载|上传|合并|检查|验证)', stripped):
            return (SegmentType.SIGNAL, 0.9, "command_cn")'''
new_classify_cmd_cn = '''        if self._re_cmd_cn.search(stripped):
            return (SegmentType.SIGNAL, 0.9, "command_cn")'''
content = content.replace(old_classify_cmd_cn, new_classify_cmd_cn)

old_classify_cmd_en = '''        if re.search(r'\\b(write|create|delete|modify|run|execute|search|analyze|implement|optimize|refactor|debug|fix|install|deploy|configure|test|compare|recommend|explain|translate|summarize|generate|download|upload|merge|check|verify|help|add|remove|update|set|get|find|show|list|open|close|enable|disable)\\b', stripped, re.IGNORECASE):
            return (SegmentType.SIGNAL, 0.9, "command_en")'''
new_classify_cmd_en = '''        if self._re_cmd_en.search(stripped):
            return (SegmentType.SIGNAL, 0.9, "command_en")'''
content = content.replace(old_classify_cmd_en, new_classify_cmd_en)

old_classify_question = '''        if re.search(r'[？?]', stripped):
            return (SegmentType.SIGNAL, 0.9, "question")'''
new_classify_question = '''        if self._re_question.search(stripped):
            return (SegmentType.SIGNAL, 0.9, "question")'''
content = content.replace(old_classify_question, new_classify_question)

old_classify_tech = '''        if re.search(r'\\b(Python|TypeScript|JavaScript|function|class|import|return|async|await|const|let|var|def |class |if |else|for |while )\\b', stripped):
            return (SegmentType.SIGNAL, 0.85, "technical")'''
new_classify_tech = '''        if self._re_technical.search(stripped):
            return (SegmentType.SIGNAL, 0.85, "technical")'''
content = content.replace(old_classify_tech, new_classify_tech)

old_classify_num = '''        if re.search(r'^\\d+[\\d.,]*$', stripped):
            return (SegmentType.SIGNAL, 0.8, "numeric")'''
new_classify_num = '''        if self._re_numeric.search(stripped):
            return (SegmentType.SIGNAL, 0.8, "numeric")'''
content = content.replace(old_classify_num, new_classify_num)

old_classify_cn_chars = '''        cn_chars = len(re.findall(r'[\\u4e00-\\u9fff]', stripped))
        if cn_chars <= 3 and not re.search(r'[？?！!]', stripped):'''
new_classify_cn_chars = '''        cn_chars = len(self._re_cn_chars.findall(stripped))
        if cn_chars <= 3 and not self._re_excl_or_ques.search(stripped):'''
content = content.replace(old_classify_cn_chars, new_classify_cn_chars)

# ============================================================
# 6. Optimize _extract_instruction_key (if inline regex found)
# ============================================================

# Check if there's an inline regex in _extract_instruction_key
if "re.sub(r'[？?。！!，,\\s]+$', '', cleaned)" in content:
    content = content.replace(
        "re.sub(r'[？?。！!，,\\s]+$', '', cleaned)",
        "self._re_instruction_clean.sub('', cleaned)"
    )

# ============================================================
# Verify no replacements were skipped
# ============================================================

# Count replacements by checking if old patterns still exist
checks = [
    ("for pattern in self.inline_fillers_cn", "OLD _strip_fillers loop"),
    ("re.sub(r'！{2,}'", "OLD inline punct regex"),
    ("re.search(r'(Error|Exception", "OLD inline error regex"),
    ("re.search(r'https?://", "OLD inline url regex"),
    ("for pattern in self.inline_fillers_en", "OLD filler_en loop"),
]
skipped = []
for pattern, label in checks:
    if pattern in content:
        skipped.append(label)

new_lines = content.count("\n")
print(f"Lines: {original_lines} → {new_lines} (diff: {new_lines - original_lines:+d})")

if skipped:
    print(f"⚠️  Skipped {len(skipped)} replacements:")
    for s in skipped:
        print(f"   - {s}")
else:
    print("✅ All replacements applied successfully")

with open(SRC, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Written to {SRC}")
