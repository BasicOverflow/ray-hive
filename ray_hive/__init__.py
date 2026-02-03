"""
Ray Hive - Distributed LLM serving engine for Ray clusters.

This package provides:
- RayHive: Main client for distributed LLM serving
- Inference functions: Standalone inference functions
- Core components: VRAMAllocator, VLLMModel, ModelOrchestrator
"""

from .client import RayHive
from .inference import (
    inference,
    a_inference,
    inference_batch,
    a_inference_batch,
)
from .core import VRAMAllocator, get_vram_allocator, VLLMModel, ModelOrchestrator
from .shutdown import shutdown_all, shutdown_model

__all__ = [
    # Main client
    "RayHive",
    # Inference functions
    "inference",
    "a_inference",
    "inference_batch",
    "a_inference_batch",
    # Core components
    "VRAMAllocator",
    "get_vram_allocator",
    "VLLMModel",
    "ModelOrchestrator",
    # Shutdown functions
    "shutdown_all",
    "shutdown_model",
]

__version__ = "0.1.0"

