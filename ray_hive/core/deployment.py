"""
Deployment service — singleton Ray actor that serializes deploy/shutdown.

DeployService runs the full per-model pipeline: HF config → planner → GPU
selection → replica reservation → parallel RayLLMActor deploy → ModelRouter.
"""
import time

import ray
from ray import serve

from .gpu_registry import get_gpu_registry
from .model_specs.planner import build_vram_reqs, plan_deployment


def _gpu_resource_name(gpu_key: str) -> str:
    """Return Ray custom resource name for a GPU key (node_gpuN)."""
    node_name, gpu_id_str = gpu_key.split(":")
    gpu_id = gpu_id_str.replace("gpu", "")
    return f"{node_name}_gpu{gpu_id}"


def _deployment_name(model_id: str, gpu_key: str) -> str:
    """Return Serve app/deployment name for a model replica on a GPU."""
    return f"{model_id}-{gpu_key.replace(':', '-').replace('_', '-')}"


def _gpu_info_entry(gpu_key: str, gpu_info: dict) -> dict:
    """Build scheduling info dict from a registry GPU view."""
    return {
        "gpu_key": gpu_key,
        "resource_name": _gpu_resource_name(gpu_key),
        "gpu_id": gpu_key.split(":")[1].replace("gpu", ""),
        "total_gb": gpu_info["total"],
        "free_gb": gpu_info["free"],
        "available_gb": gpu_info["available"],
    }


def _eligible_gpus(gpu_map: dict, min_vram_gb: float) -> list[dict]:
    """Filter GPUs with enough available VRAM for model weights."""
    eligible = []
    for gpu_key, gpu_info in gpu_map.items():
        if gpu_info["available"] < min_vram_gb:
            continue
        eligible.append(_gpu_info_entry(gpu_key, gpu_info))
    return eligible


def _resolve_pinned_gpu(gpu_map: dict, gpu_key: str, min_vram_gb: float) -> dict:
    """Resolve an explicit GPU pin from the full registry (not the filtered eligible list)."""
    if gpu_key not in gpu_map:
        raise ValueError(f"GPU {gpu_key} not in registry. Known: {sorted(gpu_map)}")
    gpu_info = gpu_map[gpu_key]
    if gpu_info["available"] < min_vram_gb:
        raise ValueError(
            f"GPU {gpu_key} has {gpu_info['available']:.2f}GB available, "
            f"need {min_vram_gb:.2f}GB for model weights"
        )
    return _gpu_info_entry(gpu_key, gpu_info)


def _resolve_replicas(gpu_map: dict, available: list[dict], replicas: int, gpu, min_vram_gb: float) -> list[dict]:
    """Pick target GPUs from available pool based on replicas count or explicit gpu pin."""
    if gpu is not None:
        if isinstance(gpu, str):
            return [_resolve_pinned_gpu(gpu_map, gpu, min_vram_gb)]
        if isinstance(gpu, list):
            return [_resolve_pinned_gpu(gpu_map, g, min_vram_gb) for g in gpu]
        raise ValueError("gpu must be a string, list of strings, or None")

    if replicas == -1:
        return available
    return available[:replicas]


@ray.remote(num_gpus=0.01)
def fetch_hf_config_dict(model_name: str) -> dict:
    """Load HF config on a GPU worker where transformers/vllm are installed."""
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(model_name, trust_remote_code=True).to_dict()


@ray.remote
def deploy_single(
    replica_id: str,
    model_id: str,
    target_gpu_id: str,
    engine_kwargs: dict,
    gpu_fraction: float,
    resource_name: str,
    route_prefix: str,
):
    """Deploy one RayLLMActor replica on a GPU worker (imports vllm here only)."""
    from ray_hive.core.ray_llm_actor import RayLLMActor

    deployment = RayLLMActor.options(
        name=replica_id,
        ray_actor_options={
            "num_gpus": gpu_fraction,
            "memory": 2 * 1024 * 1024 * 1024,
            "resources": {resource_name: 0.01},
        },
        autoscaling_config=None,
        num_replicas=1,
    ).bind(model_id=model_id, target_gpu_id=target_gpu_id, engine_kwargs=engine_kwargs)
    serve.run(deployment, name=replica_id, route_prefix=route_prefix)
    return True


@ray.remote
def deploy_router(
    model_id: str,
    model_name: str,
    gpu_deployment_names: list[str],
    replica_metadata: dict,
    resource_name: str,
):
    """Bind/run ModelRouter on a GPU worker (needs transformers/vllm)."""
    from ray_hive.core.model_router import ModelRouter

    router = ModelRouter.options(
        name=f"{model_id}-router",
        autoscaling_config=None,
        num_replicas=1,
        ray_actor_options={"num_cpus": 0.1, "resources": {resource_name: 0.01}},
    ).bind(
        model_id=model_id,
        model_name=model_name,
        gpu_deployment_names=gpu_deployment_names,
        replica_metadata=replica_metadata,
    )
    serve.run(router, name=model_id, route_prefix=f"/{model_id}")
    return True


@ray.remote(num_cpus=0)
class DeployService:
    """Singleton actor — serializes deploy/shutdown across apps."""

    def deploy_models(self, model_configs: dict, vllm_kwargs: dict | None = None) -> dict:
        """Deploy one or more models; returns per-replica plan dicts."""
        serve.start()
        vllm_kwargs = vllm_kwargs or {}
        results = {}
        for model_id, config in model_configs.items():
            results[model_id] = self._deploy_model(model_id, config, vllm_kwargs.get(model_id, {}))
        return results


    def shutdown_model(self, model_id: str) -> dict:
        """Delete all Serve apps and registry state for a model."""
        apps = serve.status().applications
        for app_name in list(apps.keys()):
            if app_name == model_id or app_name.startswith(f"{model_id}-"):
                serve.delete(name=app_name)
        registry = get_gpu_registry()
        ray.get(registry.clear_by_prefix.remote(f"{model_id}-"))
        ray.get(registry.release_deployment.remote(model_id))
        time.sleep(3.0)
        return {"model_id": model_id, "status": "shutdown"}


    def shutdown_all(self) -> dict:
        """Shutdown all Serve apps and clear registry state."""
        if serve.status().applications:
            serve.shutdown()
        registry = get_gpu_registry()
        ray.get(registry.clear_all.remote())
        time.sleep(5.0)
        return {"status": "shutdown_all"}


    def _deploy_model(self, model_id: str, config: dict, model_vllm_kwargs: dict) -> dict:
        """Run the full deploy pipeline for one model."""
        if "max_input_prompt_length" not in config or "max_output_prompt_length" not in config:
            raise ValueError("max_input_prompt_length and max_output_prompt_length are required")

        registry = get_gpu_registry()
        hf_params = ray.get(fetch_hf_config_dict.remote(config["name"]))
        vram_reqs = build_vram_reqs(hf_params, **model_vllm_kwargs)

        gpu_map = ray.get(registry.get_all_gpus.remote())
        min_vram_gb = vram_reqs.calc_weights_gb()
        available_gpus = _eligible_gpus(gpu_map, min_vram_gb)
        target_gpus = _resolve_replicas(
            gpu_map, available_gpus, config.get("replicas", -1), config.get("gpu"), min_vram_gb
        )

        max_model_len = config["max_input_prompt_length"] + config["max_output_prompt_length"]
        swap_space = float(config.get("swap_space_per_instance") or 0)
        batched_tokens_override = config.get("max_num_batched_tokens")
        max_num_seqs_override = config.get("max_num_seqs")

        replica_jobs = []
        gpu_mapping = {}
        replica_metadata = {}
        deployment_id = model_id

        for gpu_info in target_gpus:
            gpu_key = gpu_info["gpu_key"]
            used_vram = ray.get(registry.used_vram_gb.remote(gpu_key))
            vram_budget_gb = (gpu_info["available_gb"] - used_vram) * 0.95
            plan = plan_deployment(
                vram_reqs,
                vram_budget_gb=vram_budget_gb,
                live_total_vram_gb=gpu_info["total_gb"],
                max_model_len=max_model_len,
                input_len=config["max_input_prompt_length"],
                output_len=config["max_output_prompt_length"],
                max_num_batched_tokens_override=batched_tokens_override,
                max_num_seqs_override=max_num_seqs_override,
            )

            replica_id = _deployment_name(model_id, gpu_key)
            ray.get(registry.reserve_replica.remote(replica_id, gpu_key, plan["total_vram_gb"]))

            engine_kwargs = {
                "model": config["name"],
                "max_model_len": max_model_len,
                "max_num_seqs": plan["max_num_seqs"],
                "max_num_batched_tokens": plan["max_num_batched_tokens"],
                "gpu_memory_utilization": plan["gpu_memory_utilization"],
                "swap_space": swap_space,
                "enforce_eager": False,
                **model_vllm_kwargs,
            }

            gpu_fraction = max(0.01, round(plan["total_vram_gb"] / gpu_info["total_gb"], 2))
            replica_jobs.append({
                "replica_id": replica_id,
                "model_id": model_id,
                "target_gpu_id": gpu_info["gpu_id"],
                "engine_kwargs": engine_kwargs,
                "gpu_fraction": gpu_fraction,
                "resource_name": gpu_info["resource_name"],
            })
            gpu_mapping[replica_id] = gpu_key
            replica_metadata[replica_id] = plan

        if not replica_jobs:
            raise ValueError(f"No eligible GPUs for model {model_id}")

        deploy_futures = []
        for job in replica_jobs:
            replica_id = job["replica_id"]
            future = deploy_single.options(resources={job["resource_name"]: 0.01}).remote(
                replica_id,
                job["model_id"],
                job["target_gpu_id"],
                job["engine_kwargs"],
                job["gpu_fraction"],
                job["resource_name"],
                f"/{replica_id}",
            )
            deploy_futures.append(future)
        ray.get(deploy_futures)

        replica_gpu_vram_gb = {
            replica_id: {gpu_mapping[replica_id]: replica_metadata[replica_id]["total_vram_gb"]}
            for replica_id in replica_metadata
        }
        ray.get(registry.reserve_deployment.remote(deployment_id, replica_gpu_vram_gb, deployment_type="model", model_id=model_id))

        for replica_id in replica_metadata:
            ray.get(registry.mark_initialized.remote(replica_id, gpu_mapping[replica_id]))

        gpu_deployment_names = [job["replica_id"] for job in replica_jobs]
        router_resource = replica_jobs[0]["resource_name"]
        ray.get(deploy_router.options(resources={router_resource: 0.01}).remote(
            model_id,
            config["name"],
            gpu_deployment_names,
            replica_metadata,
            router_resource,
        ))

        return {
            replica_id: {"plan": replica_metadata[replica_id], "gpu_key": gpu_mapping[replica_id]}
            for replica_id in gpu_deployment_names
        }


def get_deploy_service():
    """Get or create the detached DeployService singleton actor."""
    try:
        return ray.get_actor("deploy_service", namespace="system")
    except ValueError:
        return DeployService.options(name="deploy_service", namespace="system", lifetime="detached").remote()
