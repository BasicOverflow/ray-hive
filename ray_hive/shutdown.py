"""
Shutdown API — thin wrappers that delegate to DeployService.

Serialized with deploys so shutdown cannot race with an in-flight deploy.
"""
import ray

from .core.deployment import get_deploy_service


def shutdown_all():
    """Shutdown all Serve apps and clear registry state."""
    ray.get(get_deploy_service().shutdown_all.remote())


def shutdown_model(model_id: str):
    """Shutdown one model and clear its registry reservations."""
    ray.get(get_deploy_service().shutdown_model.remote(model_id))


def kill_gpu_registry():
    """Kill detached singleton actors so they are recreated fresh on next init."""
    try:
        ray.kill(ray.get_actor("gpu_registry", namespace="system"))
    except ValueError:
        pass
    try:
        ray.kill(ray.get_actor("deploy_service", namespace="system"))
    except ValueError:
        pass
