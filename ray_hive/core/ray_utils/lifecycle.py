"""Deploy lifecycle helpers — shutdown and singleton actor teardown."""
import ray

from ray_hive.errors import ConfigError, ModelAlreadyDeployedError


def shutdown_all():
    """Shutdown all Serve apps and clear registry state (from caller / driver)."""
    from ray import serve

    from ray_hive.core.gpu_registry import get_gpu_registry

    if serve.status().applications:
        serve.shutdown()
    registry = get_gpu_registry()
    ray.get(registry.clear_all.remote())


def shutdown_model(model_id: str):
    """Shutdown one model and clear its registry reservations."""
    from ray import serve

    from ray_hive.core.gpu_registry import get_gpu_registry

    registry = get_gpu_registry()
    deployment = ray.get(registry.get_deployment.remote(model_id))
    replica_ids = list(deployment["replicas"].keys()) if deployment else []

    apps = serve.status().applications or {}
    # Replicas + registry first — if this runs inside the router (idle_timeout),
    # deleting the router app last so cleanup is not cut off mid-flight.
    for app_name in replica_ids:
        if app_name in apps:
            serve.delete(name=app_name)

    if replica_ids:
        ray.get(registry.clear_replicas.remote(replica_ids))
    ray.get(registry.release_deployment.remote(model_id))

    if model_id in (serve.status().applications or {}):
        serve.delete(name=model_id)


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

    from ray_hive.core.openai_gateway import GATEWAY_APP

    if model_id == GATEWAY_APP:
        raise ConfigError(f"model_id {model_id!r} is reserved for the OpenAI gateway")

    apps = serve.status().applications or {}
    if model_id in apps:
        raise ModelAlreadyDeployedError(f"model_id {model_id!r} already has a Serve application")
    if ray.get(registry.has_deployment.remote(model_id)):
        raise ModelAlreadyDeployedError(f"model_id {model_id!r} already registered in gpu registry")
