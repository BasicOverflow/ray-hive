"""Core Ray Hive components — deploy service, GPU registry, actors, router."""
from .gpu_registry import VRAMAllocator, get_gpu_registry

__all__ = [
    "DeployService",
    "get_deploy_service",
    "VRAMAllocator",
    "get_gpu_registry",
    "RayLLMActor",
    "ModelRouter",
]


def __getattr__(name):
    if name == "DeployService":
        from .deployment import DeployService
        return DeployService
    if name == "get_deploy_service":
        from .deployment import get_deploy_service
        return get_deploy_service
    if name == "RayLLMActor":
        from .ray_llm_actor import RayLLMActor
        return RayLLMActor
    if name == "ModelRouter":
        from .model_router import ModelRouter
        return ModelRouter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
