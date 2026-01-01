"""
VRAM Scheduler - Dynamic VRAM-aware scheduling for Ray.

This package provides:
- VRAMAllocator: Global state actor for VRAM tracking
- VLLMModel: vLLM model actor with VRAM reservation
- ModelOrchestrator: Declarative model deployment
"""

from .vram_allocator import VRAMAllocator, get_vram_allocator
from .vllm_model_actor import VLLMModel
from .model_orchestrator import ModelOrchestrator, MODELS

__all__ = [
    "VRAMAllocator",
    "get_vram_allocator",
    "VLLMModel",
    "ModelOrchestrator",
    "MODELS",
]

