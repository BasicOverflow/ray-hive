"""
Deployment service — singleton Ray actor that serializes deploy/shutdown.

DeployService runs the full per-model pipeline: HF config → planner → GPU
selection → replica reservation → parallel RayLLMActor deploy → ModelRouter.
"""
import ray
from ray import serve

from .gpu_registry import get_gpu_registry
from .gpu_alloc import assert_cpu_ram_tp_allowed, resolve_group_cpu_spill
from .model_specs.planner import normalize_hf_config, plan_deployment
from .ray_utils import (
    assert_model_id_free,
    chunk_gpu_groups,
    deployment_name,
    fixed_non_kv_gb,
    gpu_budget_frac,
    host_memory_available_gb,
    replicas_per_host,
    resolve_target_gpus,
)


@ray.remote(num_gpus=0.01)
def fetch_hf_config_dict(model_name: str) -> dict:
    """Load HF config.json on a GPU worker (no AutoConfig — works for new model_types)."""
    import json
    from pathlib import Path

    local = Path(model_name) / "config.json"
    if local.is_file():
        return json.loads(local.read_text())

    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=model_name, filename="config.json")
    return json.loads(Path(path).read_text())


@ray.remote
def deploy_single(
    replica_id: str,
    model_id: str,
    target_gpu_id: str,
    engine_kwargs: dict,
    resource_names: list[str],
    route_prefix: str,
    memory_bytes: int,
):
    """Deploy one RayLLMActor replica on a GPU worker (imports vllm here only)."""
    from ray_hive.core.ray_llm_actor import RayLLMActor

    resources = {name: 0.01 for name in resource_names}
    # num_gpus>0 makes Ray remap CUDA_VISIBLE_DEVICES and breaks custom-resource pinning
    # on multi-GPU nodes (wrong card → OOM / Serve "Failed to update"). Pin via CVD only.
    env_vars = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": target_gpu_id,
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
    }
    if len(resource_names) > 1:
        env_vars["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
    deployment = RayLLMActor.options(
        name=replica_id,
        graceful_shutdown_timeout_s=0,
        ray_actor_options={
            "num_gpus": 0,
            "memory": memory_bytes,
            "resources": resources,
            "runtime_env": {"env_vars": env_vars},
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
    chat_template_kwargs: dict | None = None,
    idle_timeout: int = -1,
):
    """Bind/run ModelRouter on a GPU worker (needs transformers/vllm)."""
    from ray_hive.core.model_router import ModelRouter

    router = ModelRouter.options(
        name=f"{model_id}-router",
        graceful_shutdown_timeout_s=0,
        autoscaling_config=None,
        num_replicas=1,
        ray_actor_options={"num_cpus": 0.1, "resources": {resource_name: 0.01}},
    ).bind(
        model_id=model_id,
        model_name=model_name,
        gpu_deployment_names=gpu_deployment_names,
        replica_metadata=replica_metadata,
        chat_template_kwargs=chat_template_kwargs or {},
        idle_timeout=idle_timeout,
    )
    serve.run(router, name=model_id, route_prefix=f"/{model_id}")
    return True


@ray.remote(num_cpus=0)
class DeployService:
    """Singleton actor — serializes deploy/shutdown across apps."""

    def deploy_models(self, model_configs: dict, vllm_kwargs: dict | None = None) -> dict:
        """Deploy one or more models; returns per-replica plan dicts."""
        serve.start(
            proxy_location="HeadOnly",
            http_options={"host": "0.0.0.0", "port": 8000},
        )
        vllm_kwargs = vllm_kwargs or {}
        results = {}
        seen = set()
        for model_id, config in model_configs.items():
            if model_id in seen:
                raise ValueError(f"duplicate model_id {model_id!r} in deploy_models batch")
            seen.add(model_id)
            results[model_id] = self._deploy_model(model_id, config, vllm_kwargs.get(model_id, {}))
        return results


    def shutdown_model(self, model_id: str) -> dict:
        """Delete this model's router + replica apps and registry state (exact ids only)."""
        from ray_hive.core.ray_utils.lifecycle import shutdown_model as _shutdown_model

        _shutdown_model(model_id)
        return {"model_id": model_id, "status": "shutdown"}


    def shutdown_all(self) -> dict:
        """Shutdown all Serve apps and clear registry state."""
        from ray_hive.core.ray_utils.lifecycle import shutdown_all as _shutdown_all

        _shutdown_all()
        return {"status": "shutdown_all"}


    def _deploy_model(self, model_id: str, config: dict, model_vllm_kwargs: dict) -> dict:
        """Run the full deploy pipeline for one model."""
        if "max_input_prompt_length" not in config or "max_output_prompt_length" not in config:
            raise ValueError("max_input_prompt_length and max_output_prompt_length are required")

        # Serve-frontend only — not valid on LLM()/EngineArgs; applied by ModelRouter.
        model_vllm_kwargs = dict(model_vllm_kwargs)
        chat_template_kwargs = model_vllm_kwargs.pop("default_chat_template_kwargs", None) or {}
        model_vllm_kwargs.pop("tensor_parallel_size", None)
        model_vllm_kwargs.pop("distributed_executor_backend", None)
        model_vllm_kwargs.pop("kv_offloading_size", None)
        model_vllm_kwargs.pop("kv_offloading_backend", None)
        model_vllm_kwargs.pop("cpu_offload_gb", None)

        registry = get_gpu_registry()
        assert_model_id_free(model_id, registry)

        hf_params = normalize_hf_config(ray.get(fetch_hf_config_dict.remote(config["name"])))
        gpu_map = ray.get(registry.get_all_gpus.remote())
        cpu_ram_cfg = float(config.get("cpu_ram_per_instance") or 0)
        tp_size, target_gpus, vram_reqs = resolve_target_gpus(
            gpu_map,
            config.get("replicas", -1),
            config.get("gpu"),
            hf_params,
            config.get("allocation_cls"),
            config.get("attention_cls"),
            model_vllm_kwargs,
            cpu_ram_cfg=cpu_ram_cfg,
        )
        assert_cpu_ram_tp_allowed(tp_size, cpu_ram_cfg)
        gpu_groups = chunk_gpu_groups(target_gpus, tp_size)
        per_host = replicas_per_host(gpu_groups)

        max_model_len = config["max_input_prompt_length"] + config["max_output_prompt_length"]
        batched_tokens_override = config.get("max_num_batched_tokens")
        max_num_seqs_override = config.get("max_num_seqs")
        weight_need = fixed_non_kv_gb(vram_reqs)

        replica_jobs = []
        gpu_mapping = {}
        replica_metadata = {}
        deployment_id = model_id

        for group in gpu_groups:
            gpu_keys = [g["gpu_key"] for g in group]
            bottleneck = min(group, key=lambda g: g["available_gb"])
            avail = min(g["available_gb"] for g in group)
            host = group[0]["gpu_key"].split(":")[0]
            if tp_size > 1:
                per_gpu_budget = avail * gpu_budget_frac(tp_size)
                cpu_ram = cpu_offload = kv_offload = 0.0
                memory_bytes = 0
            else:
                per_gpu_budget, cpu_ram, cpu_offload, kv_offload, memory_bytes = resolve_group_cpu_spill(
                    weight_need, avail, cpu_ram_cfg, host_memory_available_gb(host), per_host[host]
                )
            plan = plan_deployment(
                vram_reqs,
                vram_budget_gb=per_gpu_budget,
                live_total_vram_gb=bottleneck["total_gb"],
                max_model_len=max_model_len,
                input_len=config["max_input_prompt_length"],
                output_len=config["max_output_prompt_length"],
                max_num_batched_tokens_override=batched_tokens_override,
                max_num_seqs_override=max_num_seqs_override,
                cpu_kv_offload_gb=kv_offload,
                cpu_weight_offload_gb=cpu_offload,
            )
            plan = dict(plan)
            plan["tensor_parallel_size"] = tp_size
            plan["weights_gb"] = vram_reqs.calc_weights_gb() * tp_size
            plan["weight_need_gb"] = weight_need
            plan["cpu_ram_budget_gb"] = cpu_ram

            replica_id = deployment_name(model_id, gpu_keys)
            for gpu_info in group:
                if gpu_info["available_gb"] < plan["total_vram_gb"]:
                    raise ValueError(
                        f"GPU {gpu_info['gpu_key']} has {gpu_info['available_gb']:.2f}GB available, "
                        f"need {plan['total_vram_gb']:.2f}GB for TP plan"
                    )
                ray.get(registry.reserve_replica.remote(
                    replica_id, gpu_info["gpu_key"], plan["total_vram_gb"]
                ))

            engine_kwargs = {
                "model": config["name"],
                "max_model_len": max_model_len,
                "max_num_seqs": plan["max_num_seqs"],
                "max_num_batched_tokens": plan["max_num_batched_tokens"],
                "gpu_memory_utilization": plan["gpu_memory_utilization"],
                "enforce_eager": False,
                **model_vllm_kwargs,
            }
            if kv_offload > 0:
                engine_kwargs["kv_offloading_size"] = kv_offload
                engine_kwargs["kv_offloading_backend"] = "native"
            if cpu_offload > 0:
                engine_kwargs["cpu_offload_gb"] = cpu_offload
            if tp_size > 1:
                engine_kwargs["tensor_parallel_size"] = tp_size
                engine_kwargs["distributed_executor_backend"] = "mp"

            resource_names = [g["resource_name"] for g in group]
            replica_jobs.append({
                "replica_id": replica_id,
                "model_id": model_id,
                "target_gpu_id": ",".join(g["gpu_id"] for g in group),
                "engine_kwargs": engine_kwargs,
                "resource_names": resource_names,
                "memory_bytes": memory_bytes,
            })
            gpu_mapping[replica_id] = gpu_keys
            replica_metadata[replica_id] = plan

        if not replica_jobs:
            raise ValueError(f"No eligible GPUs for model {model_id}")

        deploy_futures = []
        for job in replica_jobs:
            resources = {name: 0.01 for name in job["resource_names"]}
            deploy_futures.append(deploy_single.options(resources=resources).remote(
                job["replica_id"],
                job["model_id"],
                job["target_gpu_id"],
                job["engine_kwargs"],
                job["resource_names"],
                f"/{job['replica_id']}",
                job["memory_bytes"],
            ))
        try:
            ray.get(deploy_futures)
        except Exception:
            for f in deploy_futures:
                ray.cancel(f, force=True)
            replica_ids = [job["replica_id"] for job in replica_jobs]
            apps = serve.status().applications or {}
            for app_name in replica_ids:
                if app_name in apps:
                    serve.delete(name=app_name)
            ray.get(registry.clear_replicas.remote(replica_ids))
            raise

        replica_gpu_vram_gb = {
            replica_id: {
                gpu_key: replica_metadata[replica_id]["total_vram_gb"]
                for gpu_key in gpu_mapping[replica_id]
            }
            for replica_id in replica_metadata
        }
        ray.get(registry.reserve_deployment.remote(
            deployment_id, replica_gpu_vram_gb, deployment_type="model", model_id=model_id
        ))

        for replica_id, gpu_keys in gpu_mapping.items():
            for gpu_key in gpu_keys:
                ray.get(registry.mark_initialized.remote(replica_id, gpu_key))

        gpu_deployment_names = [job["replica_id"] for job in replica_jobs]
        router_resource = replica_jobs[0]["resource_names"][0]
        ray.get(deploy_router.options(resources={router_resource: 0.01}).remote(
            model_id,
            config["name"],
            gpu_deployment_names,
            replica_metadata,
            router_resource,
            chat_template_kwargs,
            config.get("idle_timeout", -1),
        ))

        return {
            replica_id: {
                "plan": replica_metadata[replica_id],
                "gpu_key": gpu_mapping[replica_id][0] if len(gpu_mapping[replica_id]) == 1 else gpu_mapping[replica_id],
                "gpu_keys": gpu_mapping[replica_id],
            }
            for replica_id in gpu_deployment_names
        }


def get_deploy_service():
    """Get or create the detached DeployService singleton actor."""
    try:
        return ray.get_actor("deploy_service", namespace="system")
    except ValueError:
        return DeployService.options(name="deploy_service", namespace="system", lifetime="detached").remote()
