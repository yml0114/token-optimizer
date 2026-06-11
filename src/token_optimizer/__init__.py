"""Token Optimizer — Universal LLM Token Optimization Engine."""

from token_optimizer.api import compress_text
from token_optimizer.client import TokenOptimizer
from token_optimizer.config import OptimizerConfig
from token_optimizer.production import ProductionOptimizer, ProductionOptimizerConfig, RolloutGate
from token_optimizer.types import CompressionQuality, CompressionResult, CompressionStats

__version__ = "0.2.0"
__all__ = [
    "TokenOptimizer",
    "OptimizerConfig",
    "ProductionOptimizer",
    "ProductionOptimizerConfig",
    "RolloutGate",
    "compress_text",
    "CompressionQuality",
    "CompressionResult",
    "CompressionStats",
]
