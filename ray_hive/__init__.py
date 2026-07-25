"""
Ray Hive - Distributed LLM serving engine for Ray clusters.
"""

from .inference import (
    inference,
    a_inference,
    inference_batch,
    a_inference_batch,
    inference_stream,
)

__all__ = [
    "RayHive",
    "inference",
    "a_inference",
    "inference_batch",
    "a_inference_batch",
    "inference_stream",
    "DeployService",
    "get_deploy_service",
    "VRAMAllocator",
    "get_gpu_registry",
    "RayLLMActor",
    "ModelRouter",
    "shutdown_all",
    "shutdown_model",
    "kill_gpu_registry",
    "estimate_vram",
]

__version__ = "0.1.0"


def __getattr__(name):
    if name == "RayHive":
        from .hive import RayHive
        return RayHive
    if name == "estimate_vram":
        from .core.model_specs.estimate import estimate_vram
        return estimate_vram
    if name in ("shutdown_all", "shutdown_model", "kill_gpu_registry"):
        from .core import ray_utils
        return getattr(ray_utils, name)
    if name in ("DeployService", "get_deploy_service", "VRAMAllocator", "get_gpu_registry", "RayLLMActor", "ModelRouter"):
        from . import core
        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
