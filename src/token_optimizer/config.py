"""Global configuration for Token Optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class OptimizerConfig:
    """Configuration for the optimization engine.

    Usage:
        config = OptimizerConfig(model="deepseek-v4-flash", api_key="sk-...")
        optimizer = TokenOptimizer(config=config)
    """

    # --- Model ---
    model: str = "deepseek-v4-flash"
    api_key: str = ""
    base_url: str = ""

    # --- L0: Prefix Structure ---
    enable_prefix_reorder: bool = True
    system_prompt_position: Literal["first", "last", "auto"] = "first"
    tools_position: Literal["after_system", "before_system", "auto"] = "after_system"

    # --- L2: Prefix Cache ---
    enable_prefix_cache: bool = True
    cache_min_prefix_tokens: int = 1024   # DeepSeek minimum cache prefix
    cache_granularity: int = 64           # DeepSeek cache storage unit

    # --- L1: Input Compression ---
    enable_input_compression: bool = True
    compression_level: str = "moderate"  # "safe" | "moderate" | "aggressive"

    # --- General ---
    max_retries: int = 2
    request_timeout: float = 120.0
    track_costs: bool = True

    def __post_init__(self):
        if not self.base_url:
            self.base_url = self._auto_base_url()

    def _auto_base_url(self) -> str:
        """Auto-detect API base URL from model name."""
        model_lower = self.model.lower()
        if "deepseek" in model_lower:
            return "https://api.deepseek.com/v1"
        if "mimo" in model_lower:
            return "https://api.xiaomimimo.com/v1"
        # Fallback: OpenAI-compatible
        return "https://api.openai.com/v1"

    @property
    def is_deepseek(self) -> bool:
        return "deepseek" in self.model.lower()

    @property
    def is_mimo(self) -> bool:
        return "mimo" in self.model.lower()
