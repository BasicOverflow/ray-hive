"""
Deployment service — singleton Ray actor that serializes model deploys.

DeployService runs the full per-model pipeline: HF config → planner → GPU
selection → replica reservation → parallel RayLLMActor deploy → ModelRouter.
"""
import ray
from ray import serve

from .gpu_alloc import reject_unsupported_host_ram_kwargs
from .gpu_registry import get_gpu_registry
from .model_specs.estimate import load_hf_config_dict
from .model_specs.planner import normalize_hf_config
from .ray_utils import assert_model_id_free
from .ray_utils.placement import plan_replica_groups


@ray.remote(num_gpus=0.01)
def fetch_hf_config_dict(model_name: str) -> dict:
    """Load HF config.json on a GPU worker (delegates to shared loader)."""
    return load_hf_config_dict(model_name)


@ray.remote
def deploy_single(
    replica_id: str,
    model_id: str,
    target_gpu_id: str,
    engine_kwargs: dict,
    resource_names: list[str],
    route_prefix: str,
):
    """Deploy one RayLLMActor replica on a GPU worker (imports vllm here only)."""
    from ray_hive.core.ray_llm_actor import RayLLMActor

    resources = {name: 0.01 for name in resource_names}
    # num_gpus>0 makes Ray remap CUDA_VISIBLE_DEVICES and breaks custom-resource pinning
    # on multi-GPU nodes (wrong card → OOM / Serve "Failed to update"). Pin via CVD only.
    # Do not set ray_actor_options memory — that is a Ray scheduling reservation and
    # pending-fails on GPU workers with little advertised heap; vLLM owns host KV/spill.
    env_vars = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": target_gpu_id,
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
    }
    if len(resource_names) > 1:
        env_vars["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
    # Default "native" OffloadingConnector mmaps CPU KV under /dev/shm (often 64MB in
    # K8s). Hive always uses SimpleCPUOffload (process RAM) via kv_transfer_config;
    # keep the env bit for any residual kv_offloading_size path.
    if engine_kwargs.get("kv_transfer_config") or engine_kwargs.get("kv_offloading_size"):
        env_vars["VLLM_USE_SIMPLE_KV_OFFLOAD"] = "1"
    deployment = RayLLMActor.options(
        name=replica_id,
        graceful_shutdown_timeout_s=0,
        ray_actor_options={
            "num_gpus": 0,
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
    sleep_timeout: int = -1,
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
        sleep_timeout=sleep_timeout,
    )
    serve.run(router, name=model_id, route_prefix=f"/{model_id}")
    return True


@ray.remote(num_cpus=0)
class DeployService:
    """Singleton actor — serializes model deploys across apps."""

    def deploy_models(self, model_configs: dict, vllm_kwargs: dict | None = None) -> dict:
        """Deploy one or more models; returns per-replica plan dicts."""
        serve.start(
            proxy_location="HeadOnly",
            http_options={"host": "0.0.0.0", "port": 8000},
        )
        vllm_kwargs = vllm_kwargs or {}
        results = {}
        for model_id, config in model_configs.items():
            results[model_id] = self._deploy_model(model_id, config, vllm_kwargs.get(model_id, {}))
        return results


    def _deploy_model(self, model_id: str, config: dict, model_vllm_kwargs: dict) -> dict:
        """Run the full deploy pipeline for one model."""
        if "max_input_prompt_length" not in config or "max_output_prompt_length" not in config:
            raise ValueError("max_input_prompt_length and max_output_prompt_length are required")

        # Serve-frontend only — not valid on LLM()/EngineArgs; applied by ModelRouter.
        model_vllm_kwargs = dict(model_vllm_kwargs)
        reject_unsupported_host_ram_kwargs(model_vllm_kwargs)
        chat_template_kwargs = model_vllm_kwargs.pop("default_chat_template_kwargs", None) or {}
        model_vllm_kwargs.pop("tensor_parallel_size", None)
        model_vllm_kwargs.pop("distributed_executor_backend", None)

        registry = get_gpu_registry()
        assert_model_id_free(model_id, registry)

        hf_params = normalize_hf_config(ray.get(fetch_hf_config_dict.remote(config["name"])))
        gpu_map = ray.get(registry.get_all_gpus.remote())
        planned = plan_replica_groups(gpu_map, config, hf_params, model_vllm_kwargs, model_id)

        replica_jobs = []
        gpu_mapping = {}
        replica_metadata = {}

        for replica_id, entry in planned.items():
            plan = entry["plan"]
            group = entry["group"]
            gpu_keys = entry["gpu_keys"]
            kv_offload = entry["kv_offload"]
            cpu_offload = entry["cpu_offload"]
            tp_size = entry["tp_size"]
            max_model_len = entry["max_model_len"]

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
            if config.get("sleep_timeout", -1) > 0:
                engine_kwargs["enable_sleep_mode"] = True
            if kv_offload > 0:
                # Avoid kv_offloading_size → OffloadingConnector (/dev/shm mmap). Use
                # SimpleCPUOffloadConnector (pinned process RAM) explicitly instead.
                engine_kwargs["enable_prefix_caching"] = True
                engine_kwargs["kv_transfer_config"] = {
                    "kv_connector": "SimpleCPUOffloadConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "cpu_bytes_to_use": int(kv_offload * (1 << 30)),
                    },
                }
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
            })
            gpu_mapping[replica_id] = gpu_keys
            replica_metadata[replica_id] = plan

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
            ))
        try:
            ray.get(deploy_futures)
        except Exception as e:
            for f in deploy_futures:
                ray.cancel(f, force=True)
            replica_ids = [job["replica_id"] for job in replica_jobs]
            apps = serve.status().applications or {}
            details = []
            for app_name in replica_ids:
                app = apps.get(app_name)
                if app is None:
                    continue
                for dep_name, dep in (app.deployments or {}).items():
                    details.append(f"{dep_name}: {dep.status} — {dep.message}")
                serve.delete(name=app_name)
            ray.get(registry.clear_replicas.remote(replica_ids))
            if details:
                raise RuntimeError("; ".join(details)) from e
            raise

        replica_gpu_vram_gb = {
            replica_id: {
                gpu_key: replica_metadata[replica_id]["total_vram_gb"]
                for gpu_key in gpu_mapping[replica_id]
            }
            for replica_id in replica_metadata
        }
        ray.get(registry.reserve_deployment.remote(
            model_id, replica_gpu_vram_gb, deployment_type="model", model_id=model_id
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
            config.get("sleep_timeout", -1),
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
