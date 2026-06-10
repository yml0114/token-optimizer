"""MiMo V2 adapter — handles MiMo-specific API quirks.

MiMo differences from standard OpenAI:
1. Reasoning tokens reported in completion_tokens_details.reasoning_tokens
2. Cache writes are currently FREE (limited time, Token Plan)
3. Cache hit requires 1024+ token prefix
4. Base URL: https://token-plan-cn.xiaomimimo.com (no /v1 suffix)
5. reasoning_content field in response for chain-of-thought
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MiMoConfig:
    """MiMo-specific API configuration."""
    base_url: str = "https://token-plan-cn.xiaomimimo.com"
    chat_endpoint: str = "/v1/chat/completions"
    model: str = "mimo-v2.5"

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}{self.chat_endpoint}"


def normalize_mimo_response(raw: dict) -> dict:
    """Normalize MiMo API response to OpenAI-compatible format.

    MiMo-specific fields:
    - reasoning_content (in choices[0].message)
    - completion_tokens_details.reasoning_tokens
    """
    result = raw.copy()

    # Normalize usage fields
    usage = result.get("usage", {})

    # Ensure standard fields exist
    if "prompt_tokens" not in usage:
        usage["prompt_tokens"] = usage.get("prompt_tokens", 0)
    if "completion_tokens" not in usage:
        usage["completion_tokens"] = usage.get("completion_tokens", 0)

    # Extract reasoning tokens (MiMo specific)
    reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    if reasoning > 0:
        usage["reasoning_tokens"] = reasoning

    result["usage"] = usage

    # Normalize choices
    for choice in result.get("choices", []):
        msg = choice.get("message", {})
        # MiMo sometimes returns reasoning_content
        # We keep it as-is for the consumer to decide
        pass

    return result


def estimate_mimo_tokens_per_request(
    system_prompt_tokens: int,
    tools_tokens: int,
    history_tokens: int,
    user_tokens: int,
    output_tokens: int,
) -> dict:
    """Estimate total tokens for a typical MiMo request.

    MiMo includes reasoning tokens in completion_tokens.
    The actual "useful" output is completion_tokens - reasoning_tokens.
    """
    input_tokens = system_prompt_tokens + tools_tokens + history_tokens + user_tokens
    reasoning_ratio = 0.3  # MiMo typically uses 30% of output for reasoning
    reasoning_tokens = int(output_tokens * reasoning_ratio)
    actual_output = output_tokens - reasoning_tokens

    return {
        "input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "actual_output_tokens": actual_output,
        "input_breakdown": {
            "system": system_prompt_tokens,
            "tools": tools_tokens,
            "history": history_tokens,
            "user": user_tokens,
        },
    }
