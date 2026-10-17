"""Deploy lifecycle helpers — hive-scoped shutdown and singleton actor teardown."""
import time

import ray

from ray_hive.core.openai_gateway import GATEWAY_APP
from ray_hive.core.ray_utils.naming import ray_namespace
from ray_hive.errors import ConfigError, ModelAlreadyDeployedError


def _serve_app_names() -> set[str]:
    from ray import serve

    return set((serve.status().applications or {}).keys())


def _hive_app_names(registry) -> set[str]:
    """Serve app names owned by hive: deployments, replica ids, gateway."""
    names = {GATEWAY_APP}
    deployments = ray.get(registry.list_deployments.remote())
    for dep_id, dep in (deployments or {}).items():
        names.add(dep_id)
        for rid in (dep.get("replicas") or {}):
            names.add(rid)
        mid = dep.get("model_id")
        if mid:
            names.add(mid)
    gpus = ray.get(registry.get_all_gpus.remote())
    for gpu in (gpus or {}).values():
        names.update((gpu.get("pending") or {}).keys())
        names.update((gpu.get("active") or {}).keys())
    return names


def reconcile_with_serve(registry, serve_app_names: set[str] | None = None) -> int:
    """Drop pending/active/deployments whose Serve apps are gone. Returns cleared count."""
    apps = serve_app_names if serve_app_names is not None else _serve_app_names()
    return ray.get(registry.reconcile_with_serve.remote(list(apps)))


def _registry():
    from ray_hive.core.gpu_registry import get_gpu_registry
    return get_gpu_registry()


def shutdown_all():
    """Shutdown all hive Serve apps and clear hive registry reservations.

    Never calls serve.shutdown() — foreign apps (e.g. Lexis Markets) stay up.
    """
    from ray import serve

    registry = _registry()
    hive_apps = _hive_app_names(registry)
    apps = serve.status().applications or {}
    for name in list(hive_apps):
        if name in apps:
            serve.delete(name=name)
    ray.get(registry.clear_all.remote())
    # Brief settle so nvidia-smi free can catch up before the next deploy.
    time.sleep(2.0)


def shutdown_model(model_id: str):
    """Shutdown one model and clear its registry reservations."""
    from ray import serve

    registry = _registry()
    deployment = ray.get(registry.get_deployment.remote(model_id))
    replica_ids = list(deployment["replicas"].keys()) if deployment else []

    apps = serve.status().applications or {}
    # If registry miss, discover orphans by model_id prefix / exact name.
    if not replica_ids:
        for app_name in list(apps):
            if app_name == model_id or app_name.startswith(f"{model_id}-"):
                replica_ids.append(app_name)

    # Replicas + registry first — if this runs inside the router (idle_timeout),
    # deleting the router app last so cleanup is not cut off mid-flight.
    for app_name in replica_ids:
        if app_name in apps and app_name != model_id:
            serve.delete(name=app_name)

    if replica_ids:
        ray.get(registry.clear_replicas.remote(
            [r for r in replica_ids if r != model_id]
        ))
    ray.get(registry.release_deployment.remote(model_id))

    if model_id in (serve.status().applications or {}):
        serve.delete(name=model_id)

    time.sleep(1.0)


def kill_gpu_registry():
    """Kill detached singleton actors so they are recreated fresh on next init."""
    ns = ray_namespace()
    try:
        ray.kill(ray.get_actor("gpu_registry", namespace=ns))
    except ValueError:
        pass
    try:
        ray.kill(ray.get_actor("deploy_service", namespace=ns))
    except ValueError:
        pass


def assert_model_id_free(model_id: str, registry) -> None:
    """Reject model_id if already present in Serve or the GPU registry."""
    from ray import serve

    if model_id == GATEWAY_APP:
        raise ConfigError(f"model_id {model_id!r} is reserved for the OpenAI gateway")

    apps = serve.status().applications or {}
    if model_id in apps:
        raise ModelAlreadyDeployedError(f"model_id {model_id!r} already has a Serve application")
    if ray.get(registry.has_deployment.remote(model_id)):
        raise ModelAlreadyDeployedError(f"model_id {model_id!r} already registered in gpu registry")
