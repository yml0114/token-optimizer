"""Tests for model configuration."""

import pytest
from token_optimizer.models.model_config import (
    get_model_profile,
    estimate_cache_savings,
    DEEPSEEK_V4_FLASH,
    MIMO_V2_5_PRO,
)


class TestGetModelProfile:
    def test_deepseek_flash(self):
        profile = get_model_profile("deepseek-v4-flash")
        assert profile.input_price_per_m == 0.14
        assert profile.cache_price_per_m == 0.0028
        assert profile.supports_chat_prefix is True

    def test_mimo_pro(self):
        profile = get_model_profile("mimo-v2.5-pro")
        assert profile.input_price_per_m == 1.00
        assert profile.cache_price_per_m == 0.20
        assert profile.supports_chat_prefix is False

    def test_alias(self):
        profile = get_model_profile("deepseek-flash")
        assert profile.input_price_per_m == 0.14

    def test_unknown_model(self):
        with pytest.raises(ValueError):
            get_model_profile("nonexistent-model")


class TestEstimateCacheSavings:
    def test_deepseek_cache_savings(self):
        """DeepSeek cache saves 98% of input cost."""
        result = estimate_cache_savings(DEEPSEEK_V4_FLASH, 80_000)
        assert result["savings_pct"] > 97.0
        assert result["savings_pct"] < 100.0

    def test_mimo_cache_savings(self):
        """MiMo cache saves 80% of input cost."""
        result = estimate_cache_savings(MIMO_V2_5_PRO, 80_000)
        assert 78.0 < result["savings_pct"] < 82.0

    def test_zero_tokens(self):
        """Zero tokens should produce zero cost."""
        result = estimate_cache_savings(DEEPSEEK_V4_FLASH, 0)
        assert result["raw_cost"] == 0
        assert result["cached_cost"] == 0
