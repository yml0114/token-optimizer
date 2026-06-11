"""Token Optimizer plugin for Hermes Agent.

Integrates token compression as an llm_request middleware to reduce
conversation history cost by 40-80% while preserving semantic fidelity.

Strategy:
  1. Rule pre-compression (noise removal, whitespace collapse)
  2. Profit-aware routing (only call cheap model if savings > cost)
  3. Smart model selection (auto-discover models, price-tier ranking, fallback chain)
  4. Circuit breaker + self-healing on failures

v2 — Smart Model Allocation:
  - Scans all Hermes custom_providers for available models
  - Auto-prices known models; infers tier for unknown models by name pattern
  - Ranks candidates by cost (cheapest first) with provider reliability bonus
  - CONCURRENT tier racing: same-tier models race in parallel (first success wins)
  - Falls back to rule-only compression if all models fail

Config via environment variables:
  TOKEN_OPTIMIZER_ENABLED        — 1/0 (default: 1)
  TOKEN_OPTIMIZER_SHADOW         — 1 for shadow mode, log only (default: 0)
  TOKEN_OPTIMIZER_CHEAP_MODEL    — force specific model (overrides auto-select)
  TOKEN_OPTIMIZER_CHEAP_BASE_URL — force API base URL
  TOKEN_OPTIMIZER_CHEAP_API_KEY  — force API key
  TOKEN_OPTIMIZER_MIN_INPUT      — min tokens to trigger compression (default: 1000)
  TOKEN_OPTIMIZER_TARGET_RATIO   — target compression ratio (default: 0.35)
  TOKEN_OPTIMIZER_KEEP_RECENT    — recent messages to keep untouched (default: 4)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────

def _env_bool(key: str, default: bool = True) -> bool:
    val = os.environ.get(key, "").strip().lower()
    return val in ("1", "true", "yes", "on") if val else default

def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key, "").strip()
    return float(val) if val else default

def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key, "").strip()
    return int(val) if val else default

ENABLED = _env_bool("TOKEN_OPTIMIZER_ENABLED", True)
SHADOW = _env_bool("TOKEN_OPTIMIZER_SHADOW", False)
MIN_INPUT_TOKENS = _env_int("TOKEN_OPTIMIZER_MIN_INPUT", 1000)
TARGET_RATIO = _env_float("TOKEN_OPTIMIZER_TARGET_RATIO", 0.35)
KEEP_RECENT = _env_int("TOKEN_OPTIMIZER_KEEP_RECENT", 4)
CONCURRENT_TIMEOUT = _env_int("TOKEN_OPTIMIZER_CONCURRENT_TIMEOUT", 12)  # per-tier timeout in seconds
CONCURRENT_TIER_SIZE = _env_int("TOKEN_OPTIMIZER_CONCURRENT_TIER_SIZE", 4)  # max candidates per tier race
LLM_MIN_TOKENS = _env_int("TOKEN_OPTIMIZER_LLM_MIN_TOKENS", 1500)  # skip LLM if old msgs below this (avoid expansion)
CACHE_MAX_SIZE = _env_int("TOKEN_OPTIMIZER_CACHE_MAX", 500)  # LRU cache max entries

# ── Model-Aware Compression ─────────────────────────────────────────────────
# Context window sizes for known models (tokens)
_MODEL_CONTEXT: Dict[str, int] = {
    "gpt-5.5": 128000, "gpt-4o": 128000, "gpt-4o-mini": 128000,
    "claude-sonnet-4-20250514": 200000, "claude-3-5-sonnet": 200000,
    "deepseek-v4-pro": 65536, "deepseek-v4-flash": 65536,
    "qwen-plus": 131072,
    "mimo-v2.5-pro": 32768, "mimo-v2.5": 32768, "mimo-v2.5-free": 32768,
    "mimo-v2-omni": 32768, "mimo-v2-pro": 32768,
}

def _context_tier(model_name: str) -> Tuple[str, int, float]:
    """Map target model to (tier_label, context_window, target_ratio).
    Bigger context → less aggressive compression (preserve detail).
    Returns (label, ctx_tokens, ratio)."""
    n = model_name.lower()
    # Exact match first
    if n in _MODEL_CONTEXT:
        ctx = _MODEL_CONTEXT[n]
    # Pattern match
    elif any(k in n for k in ("claude", "gpt-5", "gpt-4o", "qwen-plus")):
        ctx = 128000
    elif any(k in n for k in ("deepseek", "qwen")):
        ctx = 65536
    elif any(k in n for k in ("mimo", "glm", "yi-", "internlm")):
        ctx = 32768
    else:
        ctx = 32768  # conservative default

    if ctx >= 128000:
        return ("large", ctx, 0.60)   # light compression — keep more detail
    elif ctx >= 64000:
        return ("mid", ctx, 0.40)     # moderate
    else:
        return ("small", ctx, 0.30)   # aggressive — squeeze harder

# ── Async Compression Cache ─────────────────────────────────────────────────
_compression_cache: OrderedDict[str, Tuple[str, str]] = OrderedDict()  # LRU: hash -> (compressed_text, model_used)
_cache_lock = threading.Lock()
_bg_inflight: set = set()  # cache keys currently being compressed (prevent duplicates)
_bg_inflight_lock = threading.Lock()
_bg_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tokopt-bg")

# User overrides (empty = auto-detect)
OVERRIDE_MODEL = os.environ.get("TOKEN_OPTIMIZER_CHEAP_MODEL", "")
OVERRIDE_BASE_URL = os.environ.get("TOKEN_OPTIMIZER_CHEAP_BASE_URL", "")
OVERRIDE_API_KEY = os.environ.get("TOKEN_OPTIMIZER_CHEAP_API_KEY", "")

# ── Model Pricing Tiers ────────────────────────────────────────────────────
# price_per_mtok: (input $/M tokens, output $/M tokens)
# lower tier_num = cheaper = higher priority

_KNOWN_MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "mimo-v2.5":       (0.14, 0.28),
    "mimo-v2":         (0.14, 0.28),
    "mimo-v2-omni":    (0.14, 0.28),
    "qwen-turbo":      (0.30, 0.60),
    "deepseek-v4-flash": (0.27, 1.10),
    "deepseek-v4-flash-free": (0.0, 0.0),
    "glm-4-flash":     (0.0, 0.0),
    "mimo-v2.5-pro":   (1.00, 3.00),
    "mimo-v2-pro":     (1.00, 3.00),
    "deepseek-v4-pro": (0.50, 2.19),
    "qwen-plus":       (0.80, 2.00),
    "gpt-4o-mini":     (0.15, 0.60),
    "gpt-4o":          (2.50, 10.0),
    "claude-sonnet-4-20250514": (3.00, 15.0),
    "claude-3-5-sonnet": (3.00, 15.0),
}

def _tier_from_name(model_name: str) -> Tuple[float, int]:
    """Infer (input_price, tier_num) from model name pattern.
    Returns (estimated_input_price_per_M, tier_num).
    tier_num: 1=cheap, 2=mid, 3=expensive."""
    n = model_name.lower()
    # Free / flash / lite / mini → cheap
    if any(k in n for k in ("free", "flash", "lite", "mini", "nano", "tiny")):
        return (0.10, 1)
    # turbo / fast → mid
    if any(k in n for k in ("turbo", "fast", "rapid")):
        return (0.40, 2)
    # pro / max / ultra / opus / sonnet → expensive
    if any(k in n for k in ("pro", "max", "ultra", "opus", "sonnet", "o1", "o3")):
        return (1.00, 3)
    # Default: assume cheap (conservative — try it, don't skip it)
    return (0.14, 1)

def _model_tier(model_name: str) -> Tuple[float, int]:
    """Get pricing for a model: known table first, then infer."""
    if model_name in _KNOWN_MODEL_PRICING:
        inp, _ = _KNOWN_MODEL_PRICING[model_name]
        tier = 1 if inp <= 0.30 else (2 if inp <= 1.0 else 3)
        return (inp, tier)
    return _tier_from_name(model_name)

# ── Global State ────────────────────────────────────────────────────────────

# Populated by _auto_detect_credentials()
_candidates: List[Dict[str, Any]] = []
# candidate = {"model": str, "base_url": str, "api_key": str,
#              "input_price": float, "tier": int, "provider": str}

CHEAP_BASE_URL = ""   # Fallback for legacy code paths
CHEAP_API_KEY = ""

# ── Token Estimation ────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cn = len(re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', text))
    rest = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', '', text)
    words = len(rest.split())
    code_boost = 1.0 + min((rest.count('{') + rest.count('`')) * 0.05, 0.4)
    return int((cn * 1.8 + words * 1.3) * code_boost)

def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        total += 4
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    total += estimate_tokens(part["text"])
    return total

# ── Rule Pre-compression ────────────────────────────────────────────────────

_NOISE = [
    (re.compile(r'\n{3,}'), '\n\n'),
    (re.compile(r'^(DEBUG|TRACE|VERBOSE)\s*:.*$', re.MULTILINE), ''),
    (re.compile(r'\x1b\[[0-9;]*m'), ''),
    (re.compile(r'https?://\S{200,}'), '[long_url]'),
]

def rule_compress(text: str) -> Tuple[str, float]:
    if not text:
        return text, 1.0
    out = text
    for pat, rep in _NOISE:
        out = pat.sub(rep, out)
    out = re.sub(r' {3,}', '  ', out)
    return out, len(out) / len(text) if text else 1.0

# ── Cheap Model API Call ────────────────────────────────────────────────────

_SYS = (
    "You are a precise text compressor. Compress the conversation history "
    "preserving ALL key information: facts, decisions, code, names, numbers, "
    "URLs, file paths. Remove filler, pleasantries, repetitions. "
    "Output ONLY the compressed text."
)

def _call_single_model(
    text: str, base_url: str, api_key: str, model: str,
    max_retries: int = 2, target_ratio: float = 0.35,
) -> Optional[str]:
    """Call a single model. Returns compressed text or None on failure."""
    # Dynamic max_tokens based on target ratio
    max_out = max(200, int(len(text) * target_ratio * 1.2))
    ratio_hint = f"Aim for ~{int(target_ratio*100)}% of original length."
    sys_prompt = _SYS + " " + ratio_hint
    # Try openai SDK
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=30)
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": text},
                    ],
                    max_tokens=max_out,
                    temperature=0.1,
                )
                result = resp.choices[0].message.content
                if result and len(result.strip()) > 20:
                    return result
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                logger.debug("Token Optimizer: %s failed (openai SDK): %s", model, e)
    except ImportError:
        pass

    # Fallback: httpx
    try:
        import httpx
        for attempt in range(max_retries):
            try:
                r = httpx.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": text},
                        ],
                        "max_tokens": max_out,
                        "temperature": 0.1,
                    },
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30,
                )
                r.raise_for_status()
                result = r.json()["choices"][0]["message"]["content"]
                if result and len(result.strip()) > 20:
                    return result
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                logger.debug("Token Optimizer: %s failed (httpx): %s", model, e)
    except ImportError:
        pass

    return None


def _discover_models(base_url: str, api_key: str) -> List[str]:
    """Call /v1/models to discover available models at a provider."""
    try:
        import httpx
        r = httpx.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        models = data.get("data", [])
        return [m.get("id", "") for m in models if m.get("id")]
    except Exception as e:
        logger.debug("Token Optimizer: model discovery failed for %s: %s", base_url, e)
        return []


def _call_cheap_model_with_fallback(text: str, target_ratio: float = 0.35) -> Tuple[Optional[str], str]:
    """Try candidates with concurrent tier racing. Returns (result, model_used).
    
    Strategy:
      1. Group candidates into tiers (by cost tier)
      2. Race all candidates in a tier concurrently (first success wins)
      3. If tier fails, move to next tier
      4. Final fallback: CHEAP_BASE_URL if configured
    
    target_ratio: compression aggressiveness (0.3=aggressive, 0.6=light)
    """
    global _candidates

    if not _candidates:
        # No candidates, try CHEAP_BASE_URL directly
        if CHEAP_BASE_URL and CHEAP_API_KEY:
            result = _call_single_model(text, CHEAP_BASE_URL, CHEAP_API_KEY, "mimo-v2.5", target_ratio=target_ratio)
            return (result, "fallback") if result else (None, "")
        return None, ""

    # Group candidates by tier
    tiers: Dict[int, List[dict]] = {}
    for c in _candidates:
        tier = c.get("tier", 99)
        tiers.setdefault(tier, []).append(c)

    # Process each tier concurrently
    for tier_num in sorted(tiers.keys()):
        tier_candidates = tiers[tier_num]
        # Cap concurrent requests per tier
        batch = tier_candidates[:CONCURRENT_TIER_SIZE]
        remaining = tier_candidates[CONCURRENT_TIER_SIZE:]

        logger.info("Token Optimizer: racing tier %d (%d candidates, %ds timeout)",
                     tier_num, len(batch), CONCURRENT_TIMEOUT)

        result, model = _race_candidates(text, batch, CONCURRENT_TIMEOUT, target_ratio)
        if result:
            return result, model

        # If tier had more candidates beyond the batch, try them as fallback
        if remaining:
            logger.info("Token Optimizer: tier %d batch failed, trying %d remaining",
                         tier_num, len(remaining))
            result, model = _race_candidates(text, remaining, CONCURRENT_TIMEOUT, target_ratio)
            if result:
                return result, model

        logger.info("Token Optimizer: tier %d exhausted, moving to next tier", tier_num)

    # Final fallback: CHEAP_BASE_URL if not already tried
    if CHEAP_BASE_URL and CHEAP_API_KEY:
        already_tried = any(
            c["base_url"] == CHEAP_BASE_URL and c["api_key"] == CHEAP_API_KEY
            for c in _candidates
        )
        if not already_tried:
            logger.info("Token Optimizer: all tiers exhausted, trying CHEAP_BASE_URL fallback")
            result = _call_single_model(text, CHEAP_BASE_URL, CHEAP_API_KEY,
                                        _candidates[0]["model"] if _candidates else "mimo-v2.5",
                                        target_ratio=target_ratio)
            if result:
                return result, "fallback"

    return None, ""


def _race_candidates(
    text: str, candidates: List[dict], timeout: float, target_ratio: float = 0.35,
) -> Tuple[Optional[str], str]:
    """Race multiple candidates concurrently. First success wins."""
    if not candidates:
        return None, ""
    if len(candidates) == 1:
        # Single candidate, no need for thread pool
        c = candidates[0]
        result = _call_single_model(text, c["base_url"], c["api_key"], c["model"], target_ratio=target_ratio)
        return (result, c["model"]) if result else (None, "")

    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        futures = {
            pool.submit(
                _call_single_model, text, c["base_url"], c["api_key"], c["model"],
                2, target_ratio
            ): c["model"]
            for c in candidates
        }
        try:
            for future in as_completed(futures, timeout=timeout):
                try:
                    result = future.result(timeout=1)
                    if result:
                        # Cancel remaining
                        for f in futures:
                            f.cancel()
                        model = futures[future]
                        return result, model
                except Exception:
                    continue
        except TimeoutError:
            logger.debug("Token Optimizer: race timeout after %.1fs", timeout)
            for f in futures:
                f.cancel()

    return None, ""

# ── Circuit Breaker + Stats ─────────────────────────────────────────────────

class _Stats:
    def __init__(self):
        self.calls = 0
        self.compressed = 0
        self.input_tok = 0
        self.output_tok = 0
        self.errors = 0
        self._circuit_open_until = 0.0

    def circuit_open(self) -> bool:
        return time.time() < self._circuit_open_until

    def trip(self, secs: float = 300):
        self._circuit_open_until = time.time() + secs
        logger.warning("Token Optimizer: circuit breaker ON for %.0fs", secs)

    def record(self, inp: int, out: int, error: bool = False):
        self.calls += 1
        self.input_tok += inp
        if error:
            self.errors += 1
            if self.calls >= 10 and self.errors / self.calls > 0.3:
                self.trip()
            return
        self.compressed += 1
        self.output_tok += out

    def summary(self) -> str:
        if not self.calls:
            return "no calls"
        ratio = self.output_tok / self.input_tok * 100 if self.input_tok else 0
        return f"{self.compressed}/{self.calls} compressed, avg {ratio:.0f}%, errs {self.errors}"

_st = _Stats()

# ── Core Pipeline ───────────────────────────────────────────────────────────

def _cache_key(messages: List[Dict[str, Any]], model_tier: str = "") -> str:
    """Generate cache key from message contents + model context tier.
    Different context tiers get different cache entries."""
    parts = []
    if model_tier:
        parts.append(f"__tier:{model_tier}")
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            parts.append(f"{m.get('role','')}:{c}")
    return hashlib.md5("\n".join(parts).encode()).hexdigest()

def _bg_compress_and_cache(cache_key: str, old_text: str, after_rule: int, target_ratio: float = 0.35):
    """Background task: run LLM compression and cache the result."""
    try:
        compressed, model_used = _call_cheap_model_with_fallback(old_text, target_ratio=target_ratio)
        if compressed:
            comp_tok = estimate_tokens(compressed)
            # Only cache if compression actually reduced tokens
            if comp_tok < after_rule:
                with _cache_lock:
                    _compression_cache[cache_key] = (compressed, model_used)
                    _compression_cache.move_to_end(cache_key)
                    while len(_compression_cache) > CACHE_MAX_SIZE:
                        _compression_cache.popitem(last=False)  # evict oldest
                logger.info("Token Optimizer: bg compressed %d→%d tokens (%s), cached",
                            after_rule, comp_tok, model_used)
            else:
                logger.info("Token Optimizer: bg compression expanded %d→%d, skipping cache",
                            after_rule, comp_tok)
        else:
            logger.debug("Token Optimizer: bg compression returned empty")
    except Exception as e:
        logger.debug("Token Optimizer: bg compression error: %s", e)
    finally:
        with _bg_inflight_lock:
            _bg_inflight.discard(cache_key)

def compress_messages(
    messages: List[Dict[str, Any]],
    target_model: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compress messages with async LLM + cache.

    Flow:
      1. Cache hit? → return cached LLM-compressed result (<1ms)
      2. Rule compression (<1ms) → return immediately
      3. If old msgs >= LLM_MIN_TOKENS → launch background LLM compression → cache result
      4. Next call with same old messages → cache hit (step 1)

    target_model: the model that will receive the compressed context.
                  Used to adjust compression aggressiveness.
    """
    # Model-aware compression ratio
    tier_label, ctx_window, dyn_ratio = _context_tier(target_model)
    if not messages:
        return messages, {"skip": "empty"}

    inp_tok = estimate_messages_tokens(messages)
    report = {"input_tokens": inp_tok, "msg_count": len(messages)}

    if inp_tok < MIN_INPUT_TOKENS:
        report["skip"] = f"below_threshold ({inp_tok}<{MIN_INPUT_TOKENS})"
        return messages, report
    if _st.circuit_open():
        report["skip"] = "circuit_breaker"
        return messages, report

    # Split: system | old | recent
    system, non_sys = [], []
    for m in messages:
        (system if m.get("role") == "system" else non_sys).append(m)

    if len(non_sys) <= KEEP_RECENT:
        report["skip"] = "too_few_turns"
        return messages, report

    recent = non_sys[-KEEP_RECENT:]
    old = non_sys[:-KEEP_RECENT]

    # Step 1: Rule compression (always, <1ms)
    rule_msgs = []
    for m in old:
        c = m.get("content", "")
        if isinstance(c, str) and c:
            comp, _ = rule_compress(c)
            nm = dict(m); nm["content"] = comp
            rule_msgs.append(nm)
        else:
            rule_msgs.append(m)

    after_rule = estimate_messages_tokens(rule_msgs)
    report["after_rule_tokens"] = after_rule

    # Step 2: Check LLM cache (instant hit from previous background compression)
    ck = _cache_key(old, tier_label)
    with _cache_lock:
        cached = _compression_cache.get(ck)
        if cached:
            _compression_cache.move_to_end(ck)  # LRU: promote on hit

    if cached:
        compressed, model_used = cached
        comp_tok = estimate_tokens(compressed)
        compressed_msg = {
            "role": "user",
            "content": f"[Compressed history — {after_rule}→{comp_tok} tokens]\n\n{compressed}",
        }
        result = system + [compressed_msg] + recent
        out_tok = estimate_messages_tokens(result)
        _st.record(inp_tok, out_tok)
        report |= {
            "output_tokens": out_tok,
            "mode": "rule+cheap_model_cached",
            "ratio": out_tok / inp_tok,
            "cheap_model": model_used,
            "cache": "hit",
            "context_tier": f"{tier_label}({ctx_window}tok,r{int(dyn_ratio*100)}%)",
            "target_model": target_model,
            "stats": _st.summary(),
        }
        return result, report

    # Step 3: Profit check — no candidates available
    if not _candidates and not CHEAP_BASE_URL:
        result = system + rule_msgs + recent
        out_tok = estimate_messages_tokens(result)
        report |= {"output_tokens": out_tok, "mode": "rule_only", "ratio": out_tok / inp_tok, "context_tier": f"{tier_label}({ctx_window}tok,r{int(dyn_ratio*100)}%)", "target_model": target_model}
        return result, report

    # Step 4: Estimate savings vs cost
    cheapest_price = _candidates[0]["input_price"] if _candidates else 0.14
    saved_tokens = int(after_rule * (1 - dyn_ratio))
    savings = saved_tokens / 1e6 * 1.00
    cheap_cost = after_rule / 1e6 * cheapest_price + int(after_rule * dyn_ratio) / 1e6 * cheapest_price * 2

    if cheap_cost >= savings:
        result = system + rule_msgs + recent
        out_tok = estimate_messages_tokens(result)
        report |= {"output_tokens": out_tok, "mode": "rule_only_unprofitable", "ratio": out_tok / inp_tok, "context_tier": f"{tier_label}({ctx_window}tok,r{int(dyn_ratio*100)}%)", "target_model": target_model}
        return result, report

    # Step 5: Skip LLM if old messages too short (avoid expansion)
    if after_rule < LLM_MIN_TOKENS:
        result = system + rule_msgs + recent
        out_tok = estimate_messages_tokens(result)
        report |= {
            "output_tokens": out_tok,
            "mode": "rule_only_short",
            "ratio": out_tok / inp_tok,
            "skip_llm": f"below {LLM_MIN_TOKENS} tokens",
            "context_tier": f"{tier_label}({ctx_window}tok,r{int(dyn_ratio*100)}%)",
            "target_model": target_model,
        }
        return result, report

    # Step 6: No cache — return rule-only immediately + launch background LLM
    old_text = "\n\n".join(
        f"[{m.get('role','?')}]: {m.get('content','')}" for m in rule_msgs
    )
    with _bg_inflight_lock:
        if ck not in _bg_inflight:
            _bg_inflight.add(ck)
            _bg_pool.submit(_bg_compress_and_cache, ck, old_text, after_rule, dyn_ratio)

    result = system + rule_msgs + recent
    out_tok = estimate_messages_tokens(result)
    report |= {
        "output_tokens": out_tok,
        "mode": "rule+bg_pending",
        "ratio": out_tok / inp_tok,
        "cache": "miss_bg_started",
        "target_model": target_model or "auto",
        "context_tier": f"{tier_label}({ctx_window}tok,r{int(dyn_ratio*100)}%)",
        "stats": _st.summary(),
    }
    return result, report

# ── Shadow Evaluate ─────────────────────────────────────────────────────────

def shadow_evaluate(messages):
    """Evaluate compression without applying. Returns report only."""
    _, report = compress_messages(messages)
    report["shadow"] = True
    return report

# ── Auto-detect: Build Candidate List ───────────────────────────────────────

def _auto_detect_credentials(force: bool = False):
    """Discover all available models across Hermes providers.
    Builds _candidates list ranked by cost (cheapest first)."""
    global _candidates, CHEAP_BASE_URL, CHEAP_API_KEY

    if not force and _candidates:
        return

    _candidates = []
    cplist = []

    # Parse config.yaml
    try:
        from pathlib import Path
        config_path = Path.home() / ".hermes" / "config.yaml"
        if not config_path.exists():
            return
        text = config_path.read_text(encoding="utf-8")

        try:
            import yaml  # type: ignore
            config = yaml.safe_load(text) or {}
            cplist = config.get("custom_providers", [])
        except Exception:
            # Regex fallback
            import re as _re
            for match in _re.finditer(r'- name:\s*[\'"]?(\S+?)[\'"]?\s*$', text, _re.MULTILINE):
                idx = match.start()
                block_end = text.find('\n- name:', idx + 1)
                if block_end == -1:
                    block_end = len(text)
                block = text[idx:block_end]
                bu = _re.search(r'base_url:\s*(\S+)', block)
                ke = _re.search(r'key_env:\s*(\S+)', block)
                if bu:
                    cplist.append({
                        "name": match.group(1),
                        "base_url": bu.group(1).strip("'\""),
                        "key_env": ke.group(1).strip("'\"") if ke else "",
                    })
    except Exception as e:
        logger.debug("Token Optimizer: config parse failed: %s", e)
        return

    if not isinstance(cplist, list):
        return

    # Load .env if needed
    _load_env_file()

    # Score and process each provider
    def _provider_score(p: dict) -> int:
        if not isinstance(p, dict):
            return -999
        url = p.get("base_url", "")
        key_env = p.get("key_env", "")
        has_key = bool(os.environ.get(key_env, ""))
        if not has_key:
            return -100
        score = 0
        if "xiaomimimo" in url:
            score += 10
        if "token-plan" in url:
            score += 5
        return score

    sorted_providers = sorted(
        [p for p in cplist if isinstance(p, dict)],
        key=_provider_score,
        reverse=True,
    )

    for p in sorted_providers:
        url = p.get("base_url", "")
        key_env = p.get("key_env", "")
        api_key = os.environ.get(key_env, "")
        pname = p.get("name", "unknown")

        if not url or not api_key:
            continue

        # Get model list from config
        cfg_models = set()
        models_field = p.get("models", {})
        if isinstance(models_field, dict):
            cfg_models = set(models_field.keys())
        elif isinstance(models_field, list):
            cfg_models = set(models_field)

        # Discover actual models via API (if no env override)
        if not OVERRIDE_MODEL:
            discovered = _discover_models(url, api_key)
            if discovered:
                available = set(discovered)
                if cfg_models:
                    available &= cfg_models
            else:
                available = cfg_models
        else:
            available = {OVERRIDE_MODEL} if OVERRIDE_MODEL in cfg_models or not cfg_models else set()

        if not available:
            continue

        for model_name in available:
            if not model_name:
                continue
            # Skip non-text models (TTS/ASR/embedding/reranker)
            _skip_keywords = ("tts", "asr", "stt", "voice", "audio", "speech",
                              "embedding", "rerank", "reranker", "image", "vision",
                              "whisper", "tts-voiceclone", "tts-voicedesign")
            if any(kw in model_name.lower() for kw in _skip_keywords):
                logger.debug("Token Optimizer: skipping non-text model: %s", model_name)
                continue
            price, tier = _model_tier(model_name)
            _candidates.append({
                "model": model_name,
                "base_url": url,
                "api_key": api_key,
                "input_price": price,
                "tier": tier,
                "provider": pname,
            })

    # Set legacy globals from cheapest candidate
    if _candidates:
        CHEAP_BASE_URL = _candidates[0]["base_url"]
        CHEAP_API_KEY = _candidates[0]["api_key"]

    # If user forced a specific model, ensure it's in candidates with highest priority
    if OVERRIDE_MODEL and OVERRIDE_BASE_URL and OVERRIDE_API_KEY:
        _candidates.insert(0, {
            "model": OVERRIDE_MODEL,
            "base_url": OVERRIDE_BASE_URL,
            "api_key": OVERRIDE_API_KEY,
            "input_price": 0.0,
            "tier": 0,
            "provider": "user_override",
        })
        CHEAP_BASE_URL = OVERRIDE_BASE_URL
        CHEAP_API_KEY = OVERRIDE_API_KEY

    # Sort by (tier, price) — cheapest first
    _candidates.sort(key=lambda c: (c["tier"], c["input_price"]))

    if _candidates:
        top3 = [f"{c['model']}@{c['provider']}(t{c['tier']},${c['input_price']})" for c in _candidates[:5]]
        logger.info("Token Optimizer: %d candidates, top: %s", len(_candidates), " > ".join(top3))
    else:
        logger.warning("Token Optimizer: no model candidates found")


def _load_env_file():
    """Load .env file into os.environ (setdefault = don't overwrite)."""
    try:
        from pathlib import Path
        env_path = Path.home() / ".hermes" / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


# ── Hermes Plugin Entry Point ───────────────────────────────────────────────

def _llm_request_middleware(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """LLM request middleware — compresses messages before API call."""
    if not ENABLED:
        return None

    request = kwargs.get("request", {})
    if not isinstance(request, dict):
        return None

    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return None

    if SHADOW:
        report = shadow_evaluate(messages)
        if not report.get("skip"):
            logger.info("Token Optimizer [SHADOW]: %s", report)
        return None

    # Extract target model for model-aware compression
    target_model = request.get("model", "")
    compressed, report = compress_messages(messages, target_model=target_model)

    if report.get("skip"):
        logger.debug("Token Optimizer: skip — %s", report.get("skip"))
        return None

    logger.info(
        "Token Optimizer: %d→%d tokens (%.0f%%, %s, model=%s, ctx=%s)",
        report.get("input_tokens", 0),
        report.get("output_tokens", 0),
        report.get("ratio", 1.0) * 100,
        report.get("mode", "?"),
        report.get("cheap_model", "?"),
        report.get("context_tier", "?"),
    )

    new_req = dict(request)
    new_req["messages"] = compressed
    return {"request": new_req}


def register(ctx) -> None:
    """Hermes plugin registration."""
    _auto_detect_credentials()
    ctx.register_middleware("llm_request", _llm_request_middleware)
    models_str = ", ".join(f"{c['model']}(t{c['tier']})" for c in _candidates[:5])
    logger.info(
        "Token Optimizer v2: registered (enabled=%s shadow=%s candidates=%d [%s] min=%d ratio=%.0f%%)",
        ENABLED, SHADOW, len(_candidates), models_str, MIN_INPUT_TOKENS, TARGET_RATIO * 100,
    )
