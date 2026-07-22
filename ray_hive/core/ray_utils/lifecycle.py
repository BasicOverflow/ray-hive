"""Deploy lifecycle helpers — shutdown and singleton actor teardown."""
import ray


def shutdown_all():
    """Shutdown all Serve apps and clear registry state."""
    from ray_hive.core.deployment import get_deploy_service
    ray.get(get_deploy_service().shutdown_all.remote())


def shutdown_model(model_id: str):
    """Shutdown one model and clear its registry reservations."""
    from ray_hive.core.deployment import get_deploy_service
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


def assert_model_id_free(model_id: str, registry) -> None:
    """Reject model_id if already present in Serve or the GPU registry."""
    from ray import serve

    apps = serve.status().applications or {}
    if model_id in apps:
        raise ValueError(f"model_id {model_id!r} already has a Serve application")
    if ray.get(registry.has_deployment.remote(model_id)):
        raise ValueError(f"model_id {model_id!r} already registered in gpu registry")
