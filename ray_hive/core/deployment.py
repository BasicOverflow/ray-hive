"""
Deployment service — singleton Ray actor that serializes deploy/shutdown.

DeployService runs the full per-model pipeline: HF config → planner → GPU
selection → replica reservation → parallel RayLLMActor deploy → ModelRouter.
"""
import time

import ray
from ray import serve
from transformers import AutoConfig

from .gpu_registry import get_gpu_registry
from .model_specs.planner import build_vram_reqs, plan_deployment
from .model_router import ModelRouter


def _gpu_resource_name(gpu_key: str) -> str:
    """Return Ray custom resource name for a GPU key (node_gpuN)."""
    node_name, gpu_id_str = gpu_key.split(":")
    gpu_id = gpu_id_str.replace("gpu", "")
    return f"{node_name}_gpu{gpu_id}"


def _deployment_name(model_id: str, gpu_key: str) -> str:
    """Return Serve app/deployment name for a model replica on a GPU."""
    return f"{model_id}-{gpu_key.replace(':', '-').replace('_', '-')}"


def _eligible_gpus(gpu_map: dict, min_vram_gb: float) -> list[dict]:
    """Filter GPUs with enough available VRAM for model weights."""
    eligible = []
    for gpu_key, gpu_info in gpu_map.items():
        if len(gpu_key) > 50 or gpu_key.startswith("c"):
            continue
        if gpu_info["available"] < min_vram_gb:
            continue
        eligible.append({
            "gpu_key": gpu_key,
            "resource_name": _gpu_resource_name(gpu_key),
            "gpu_id": gpu_key.split(":")[1].replace("gpu", ""),
            "total_gb": gpu_info["total"],
            "free_gb": gpu_info["free"],
            "available_gb": gpu_info["available"],
        })
    return eligible


def _resolve_replicas(available: list[dict], replicas: int, gpu) -> list[dict]:
    """Pick target GPUs from available pool based on replicas count or explicit gpu pin."""
    if gpu is not None:
        if isinstance(gpu, str):
            picked = [g for g in available if g["gpu_key"] == gpu]
            if not picked:
                raise ValueError(f"GPU {gpu} not found or insufficient VRAM")
            return picked[:1]
        if isinstance(gpu, list):
            picked = [g for g in available if g["gpu_key"] in gpu]
            if len(picked) != len(gpu):
                missing = set(gpu) - {g["gpu_key"] for g in picked}
                raise ValueError(f"GPUs not found or insufficient VRAM: {missing}")
            return picked
        raise ValueError("gpu must be a string, list of strings, or None")

    if replicas == -1:
        return available
    return available[:replicas]


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
        registry = get_gpu_registry()
        hf_config = AutoConfig.from_pretrained(config["name"], trust_remote_code=True)
        vram_reqs = build_vram_reqs(hf_config, **model_vllm_kwargs)

        vram_weights_gb = config.get("vram_weights_gb")
        if vram_weights_gb is None:
            vram_weights_gb = vram_reqs.calc_weights_gb()

        gpu_map = ray.get(registry.get_all_gpus.remote())
        available_gpus = _eligible_gpus(gpu_map, vram_weights_gb)
        target_gpus = _resolve_replicas(available_gpus, config.get("replicas", -1), config.get("gpu"))

        max_model_len = config["max_input_prompt_length"] + config["max_output_prompt_length"]
        swap_space = float(config.get("swap_space_per_instance") or 0)
        batched_tokens_override = config.get("max_num_batched_tokens")

        replica_jobs = []
        gpu_mapping = {}
        replica_metadata = {}
        deployment_id = model_id

        for gpu_info in target_gpus:
            gpu_key = gpu_info["gpu_key"]
            used_vram = ray.get(registry.used_vram_gb.remote(gpu_key))
            plan = plan_deployment(
                vram_reqs,
                used_vram_gb=used_vram,
                live_free_vram_gb=gpu_info["free_gb"],
                live_total_vram_gb=gpu_info["total_gb"],
                max_model_len=max_model_len,
                input_len=config["max_input_prompt_length"],
                output_len=config["max_output_prompt_length"],
                max_num_batched_tokens_override=batched_tokens_override,
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
        router = ModelRouter.options(name=f"{model_id}-router", autoscaling_config=None, num_replicas=1).bind(
            model_id=model_id,
            model_name=config["name"],
            gpu_deployment_names=gpu_deployment_names,
            replica_metadata=replica_metadata,
        )
        serve.run(router, name=model_id, route_prefix=f"/{model_id}")

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
