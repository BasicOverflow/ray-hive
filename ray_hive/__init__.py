"""
Ray Hive - Distributed LLM serving engine for Ray clusters.
"""

from .hive import RayHive
from .inference import (
    inference,
    a_inference,
    inference_batch,
    a_inference_batch,
)
from .core import (
    DeployService,
    get_deploy_service,
    VRAMAllocator,
    get_gpu_registry,
    RayLLMActor,
    ModelRouter,
)
from .shutdown import shutdown_all, shutdown_model, kill_gpu_registry

__all__ = [
    "RayHive",
    "inference",
    "a_inference",
    "inference_batch",
    "a_inference_batch",
    "DeployService",
    "get_deploy_service",
    "VRAMAllocator",
    "get_gpu_registry",
    "RayLLMActor",
    "ModelRouter",
    "shutdown_all",
    "shutdown_model",
    "kill_gpu_registry",
]

__version__ = "0.1.0"
