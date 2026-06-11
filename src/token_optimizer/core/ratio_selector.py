"""Adaptive keep_ratio selector — analyze content, pick the right compression intensity.

Strategy (Quality-First, v2):
  - Density compression works great for dialogue/JSON → can use lower ratios
  - Density compression is weak for specs/code (already dense) → ratio matters more
  - Specs: ratio 0.50 (density compression won't help, just cut proportionally)
  - Code:  ratio 0.70 (quality cliff below 0.60, must preserve structure)
  - JSON:  ratio 0.50 (density compression preserves 71%+ quality at this level)
  - Dialog: ratio 0.50 (density compression preserves 58%+ quality)

Philosophy: The right ratio depends on whether density compression helps.
  - If it helps (dialogue/JSON): compress more aggressively, density scoring keeps the good stuff
  - If it doesn't (specs/code): accept the ratio's quality ceiling and compress anyway
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ContentProfile:
    """Characteristics of the input content."""
    total_tokens: int
    message_count: int
    length_bucket: str          # short / medium / long / very_long
    has_json_blocks: bool
    json_ratio: float           # fraction of tokens that are JSON
    duplicate_ratio: float      # fraction of near-duplicate segments
    role_distribution: Dict[str, int]
    avg_msg_tokens: float
    compression_urgency: float  # 0.0-1.0, how aggressively we should compress
    is_specs_like: bool         # high density structured content
    is_code_like: bool          # code-heavy content


@dataclass
class CompressionPolicy:
    """Resulting compression strategy."""
    keep_ratio: float
    json_aware: bool
    near_dedup: bool
    method: str                 # e.g. "adaptive", "json-first", "specs", "code"
    rationale: str              # human-readable explanation


# ── Thresholds ──────────────────────────────────────────────────────────────
SHORT_TOKENS = 400
MEDIUM_TOKENS = 1500
LONG_TOKENS = 4000

JSON_RATIO_THRESHOLD = 0.3   # 30%+ JSON tokens → treat as JSON-heavy
DUP_RATIO_THRESHOLD = 0.1    # 10%+ duplicates → enable dedup

# Quality-tested ratio map (updated after density compression benchmark)
RATIO_MAP = {
    "short":     0.95,   # Very short, keep almost all — IC already cleans noise
    "medium":    0.75,   # Moderate conversations
    "long":      0.60,   # Long conversations
    "very_long": 0.50,   # Deep compression
}

# Specs/code have different quality curves
SPECS_RATIO = 0.75   # High-density specs: quality-first, little safe redundancy to remove
CODE_RATIO = 0.75    # Quality cliff below 0.60 (49%→11%); 0.75 balances quality and savings


def _detect_specs_like(messages: List[Dict[str, str]]) -> bool:
    """Detect if content is specs/infrastructure/configuration-like."""
    import re
    specs_score = 0
    total_chars = 0

    for msg in messages:
        content = msg.get("content", "")
        if not content:
            continue
        total_chars += len(content)

        # Key-value patterns (": ", "= ", "→")
        specs_score += len(re.findall(r'[\w]+[:：=]\s*\S', content)) * 2

        # List markers ("- item", "• item", "1. item")
        specs_score += len(re.findall(r'^\s*[-•*]\s+\S', content, re.MULTILINE)) * 1.5

        # Numbers (specs have lots of numbers)
        specs_score += len(re.findall(r'\b\d[\d,.]*\b', content)) * 1

        # Units (GB, MB, ms, etc.)
        specs_score += len(re.findall(r'\b\d+\s*(GB|MB|ms|vCPU|req|min|sec|hour|day|month)\b', content, re.IGNORECASE)) * 3

    return specs_score > total_chars * 0.15 and total_chars > 100


def _detect_code_like(messages: List[Dict[str, str]]) -> bool:
    """Detect if content is code-heavy."""
    import re
    code_indicators = 0
    total_chars = 0

    for msg in messages:
        content = msg.get("content", "")
        if not content:
            continue
        total_chars += len(content)

        # Code blocks
        code_indicators += content.count('```') * 50

        # Function/class definitions
        code_indicators += len(re.findall(r'\b(def |class |function |const |let |var |import )', content)) * 10

        # Code patterns
        code_indicators += len(re.findall(r'[{}\[\]();]', content)) * 2

        # Indentation (4+ spaces at line start)
        code_indicators += len(re.findall(r'^    \S', content, re.MULTILINE)) * 3

    return code_indicators > total_chars * 0.1 and total_chars > 100


def profile_content(messages: List[Dict[str, str]], count_tokens_fn=None) -> ContentProfile:
    """Analyze messages and return a content profile."""
    if count_tokens_fn is None:
        count_tokens_fn = lambda t: len(t) // 4

    total = 0
    json_tokens = 0
    roles = {}
    json_indicators = ['{', '}', '[', ']', '":', 'null', 'true', 'false']

    for msg in messages:
        content = msg.get("content", "")
        tok = count_tokens_fn(content)
        total += tok

        role = msg.get("role", "unknown")
        roles[role] = roles.get(role, 0) + 1

        json_chars = sum(content.count(ind) for ind in json_indicators)
        if json_chars > len(content) * 0.05:
            json_tokens += tok

    n = len(messages)
    json_ratio = json_tokens / max(total, 1)

    # Simple duplicate detection
    dup_count = 0
    pairs = 0
    for i in range(min(n, 30)):
        for j in range(i + 1, min(n, 30)):
            pairs += 1
            ci = messages[i].get("content", "")
            cj = messages[j].get("content", "")
            if len(ci) < 20 or len(cj) < 20:
                continue
            tri_i = set(ci[k:k+3] for k in range(len(ci) - 2))
            tri_j = set(cj[k:k+3] for k in range(len(cj) - 2))
            if not tri_i or not tri_j:
                continue
            overlap = len(tri_i & tri_j) / len(tri_i | tri_j)
            if overlap > 0.6:
                dup_count += 1

    dup_ratio = dup_count / max(pairs, 1)

    # Length bucket
    if total < SHORT_TOKENS:
        bucket = "short"
    elif total < MEDIUM_TOKENS:
        bucket = "medium"
    elif total < LONG_TOKENS:
        bucket = "long"
    else:
        bucket = "very_long"

    # Compression urgency
    urgency = 0.0
    if total > 8000:
        urgency += 0.4
    elif total > 4000:
        urgency += 0.3
    elif total > 1500:
        urgency += 0.2
    if n > 30:
        urgency += 0.2
    if dup_ratio > 0.3:
        urgency += 0.2
    if json_ratio > 0.5:
        urgency += 0.1
    urgency = min(urgency, 1.0)

    # Content type detection
    is_specs = _detect_specs_like(messages)
    is_code = _detect_code_like(messages)

    return ContentProfile(
        total_tokens=total,
        message_count=n,
        length_bucket=bucket,
        has_json_blocks=json_ratio > 0.1,
        json_ratio=json_ratio,
        duplicate_ratio=dup_ratio,
        role_distribution=roles,
        avg_msg_tokens=total / max(n, 1),
        compression_urgency=urgency,
        is_specs_like=is_specs,
        is_code_like=is_code,
    )


def select_ratio(
    profile: ContentProfile,
    user_keep_ratio: Optional[float] = None,
) -> CompressionPolicy:
    """Select the optimal keep_ratio and strategy based on content profile.

    Priority order:
    1. User override
    2. JSON-heavy → json-first with lower ratios
    3. Specs-like → specs ratio (density compression weak)
    4. Code-like → code ratio (quality cliff below 0.60)
    5. Standard dialogue → adaptive by length
    """
    # ── User override ────────────────────────────────────────────────────
    if user_keep_ratio is not None:
        return CompressionPolicy(
            keep_ratio=user_keep_ratio,
            json_aware=profile.json_ratio > JSON_RATIO_THRESHOLD,
            near_dedup=profile.duplicate_ratio > DUP_RATIO_THRESHOLD,
            method="user_override",
            rationale=f"User specified keep_ratio={user_keep_ratio:.2f}"
        )

    # ── JSON-heavy content ───────────────────────────────────────────────
    if profile.json_ratio > JSON_RATIO_THRESHOLD:
        if profile.length_bucket in ("short",):
            ratio = 0.70   # Density compress now works, need higher ratio
        elif profile.length_bucket in ("medium",):
            ratio = 0.60   # Was 0.45
        else:
            ratio = 0.50   # Was 0.40
        return CompressionPolicy(
            keep_ratio=ratio,
            json_aware=True,
            near_dedup=profile.duplicate_ratio > DUP_RATIO_THRESHOLD,
            method="json-first",
            rationale=f"JSON-heavy ({profile.json_ratio:.0%}), "
                      f"{profile.length_bucket} → keep {ratio:.0%}"
        )

    # ── Specs-like content ───────────────────────────────────────────────
    if profile.is_specs_like:
        ratio = SPECS_RATIO  # 0.50
        return CompressionPolicy(
            keep_ratio=ratio,
            json_aware=profile.has_json_blocks,
            near_dedup=profile.duplicate_ratio > DUP_RATIO_THRESHOLD,
            method="specs",
            rationale=f"Specs-like (dense structured content), "
                      f"density compression ineffective → keep {ratio:.0%}"
        )

    # ── Code-like content ────────────────────────────────────────────────
    if profile.is_code_like:
        ratio = CODE_RATIO  # 0.70
        return CompressionPolicy(
            keep_ratio=ratio,
            json_aware=profile.has_json_blocks,
            near_dedup=profile.duplicate_ratio > DUP_RATIO_THRESHOLD,
            method="code",
            rationale=f"Code-like (quality cliff < 0.60), "
                      f"preserve structure → keep {ratio:.0%}"
        )

    # ── Standard content by length bucket ────────────────────────────────
    base = RATIO_MAP[profile.length_bucket]

    # Duplicate adjustment
    if profile.duplicate_ratio > 0.3:
        base = max(base - 0.10, 0.20)
    elif profile.duplicate_ratio > DUP_RATIO_THRESHOLD:
        base = max(base - 0.05, 0.20)

    return CompressionPolicy(
        keep_ratio=round(base, 2),
        json_aware=profile.has_json_blocks,
        near_dedup=profile.duplicate_ratio > DUP_RATIO_THRESHOLD,
        method="adaptive",
        rationale=(
            f"{profile.length_bucket} ({profile.total_tokens}t, "
            f"{profile.message_count}msg), "
            f"JSON:{profile.json_ratio:.0%}, "
            f"Dup:{profile.duplicate_ratio:.0%} → keep {base:.0%}"
        )
    )


def get_compression_plan(
    messages: List[Dict[str, str]],
    user_keep_ratio: Optional[float] = None,
    count_tokens_fn=None,
) -> tuple:
    """One-call convenience: profile + policy.

    Returns (ContentProfile, CompressionPolicy)
    """
    prof = profile_content(messages, count_tokens_fn)
    pol = select_ratio(prof, user_keep_ratio)
    return prof, pol
