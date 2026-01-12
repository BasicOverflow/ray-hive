"""
VRAM Scheduler - Dynamic VRAM-aware scheduling for Ray.

This package provides:
- VRAMScheduler: Main client for VRAM-aware LLM serving
- Inference functions: Standalone inference functions
- Core components: VRAMAllocator, VLLMModel, ModelOrchestrator
"""

from .client import VRAMScheduler
from .inference import (
    inference,
    a_inference,
    inference_batch,
    a_inference_batch,
    streaming_batch,
)
from .core import VRAMAllocator, get_vram_allocator, VLLMModel, ModelOrchestrator
from .shutdown import shutdown_all, shutdown_model

__all__ = [
    # Main client
    "VRAMScheduler",
    # Inference functions
    "inference",
    "a_inference",
    "inference_batch",
    "a_inference_batch",
    "streaming_batch",
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

