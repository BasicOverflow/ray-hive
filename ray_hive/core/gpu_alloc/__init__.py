"""Public API for GPU allocation policies."""
from .arch_reqs import (
    FP8_MIN_COMPUTE_CAP,
    model_needs_fp8_hardware,
    required_min_compute_cap,
)
from .base import BaseGpuAllocator, TP1_BUDGET_FRAC
from .conserve_tdp import ConserveTdpAllocator
from .performance import PerformanceAllocator
from .tensor_parallel import TensorParallelAllocator

__all__ = [
    "BaseGpuAllocator",
    "TP1_BUDGET_FRAC",
    "FP8_MIN_COMPUTE_CAP",
    "model_needs_fp8_hardware",
    "required_min_compute_cap",
    "PerformanceAllocator",
    "ConserveTdpAllocator",
    "TensorParallelAllocator",
]
