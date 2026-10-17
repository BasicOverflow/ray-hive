"""
Deployment service — singleton Ray actor that serializes model deploys.

DeployService runs the full per-model pipeline: HF config → planner → GPU
selection → replica reservation → parallel RayLLMActor deploy → ModelRouter.
"""
import time

import ray
from ray import serve

from ray_hive.errors import ConfigError, DeployError, InsufficientVramError

from .gpu_registry import get_gpu_registry
from .model_specs.estimate import load_hf_config_dict
from .model_specs.planner import normalize_hf_config
from .ray_utils import assert_model_id_free
from .ray_utils.naming import NAMESPACE_ENV, ray_namespace
from .ray_utils.placement import plan_replica_groups
from .ray_utils.session import SERVE_FASTAPI_RUNTIME_ENV

# Bump when worker-side actor/router generate behavior changes so a stale
# detached DeployService (old working_dir zip) is killed and recreated.
HIVE_DEPLOY_CODE_REV = 36


@ray.remote(num_cpus=0)
def fetch_hf_config_dict(model_name: str) -> dict:
    """Load HF config.json on a GPU worker (needs transformers)."""
    return load_hf_config_dict(model_name)


@ray.remote(num_cpus=0)
def deploy_single(
    replica_id: str,
    model_id: str,
    target_gpu_id: str,
    engine_kwargs: dict,
    resource_names: list[str],
    route_prefix: str,
    pooling: bool = False,
    multimodal: bool = False,
):
    """Deploy one RayLLMActor replica on a GPU worker (imports vllm here only)."""
    from ray_hive.core.ray_llm_actor import RayLLMActor

    resources = {name: 0.01 for name in resource_names}
    # num_gpus>0 makes Ray remap CUDA_VISIBLE_DEVICES and breaks custom-resource pinning
    # on multi-GPU nodes (wrong card → OOM / Serve "Failed to update"). Pin via CVD only.
    # Do not set ray_actor_options memory — that is a Ray scheduling reservation and
    # pending-fails on GPU workers with little advertised heap.
    env_vars = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": target_gpu_id,
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        NAMESPACE_ENV: ray_namespace(),
    }
    if len(resource_names) > 1:
        env_vars["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
    deployment = RayLLMActor.options(
        name=replica_id,
        graceful_shutdown_timeout_s=0,
        ray_actor_options={
            "num_cpus": 0,
            "num_gpus": 0,
            "resources": resources,
            "runtime_env": {"env_vars": env_vars},
        },
        autoscaling_config=None,
        num_replicas=1,
    ).bind(
        model_id=model_id,
        target_gpu_id=target_gpu_id,
        engine_kwargs=engine_kwargs,
        pooling=pooling,
        multimodal=multimodal,
    )
    serve.run(deployment, name=replica_id, route_prefix=route_prefix)
    return True


@ray.remote(num_cpus=0)
def deploy_router(
    model_id: str,
    model_name: str,
    gpu_deployment_names: list[str],
    replica_metadata: dict,
    resource_name: str,
    chat_template_kwargs: dict | None = None,
    idle_timeout: int = -1,
    sleep_timeout: int = -1,
    pooling: bool = False,
    multimodal: bool = False,
    chat_template: str | None = None,
):
    """Bind/run ModelRouter on a GPU worker (needs transformers/vllm)."""
    from ray_hive.core.model_router import ModelRouter

    # Low CPU: lexis-markets often leaves <3 CPUs free; default 1.0 never schedules.
    # Pin FastAPI so ingress unpickle matches ModelRouter (worker image / client).
    router = ModelRouter.options(
        name=f"{model_id}-router",
        graceful_shutdown_timeout_s=0,
        autoscaling_config=None,
        num_replicas=1,
        ray_actor_options={
            "num_cpus": 0,
            "resources": {resource_name: 0.01},
            "runtime_env": {
                **dict(SERVE_FASTAPI_RUNTIME_ENV),
                "env_vars": {NAMESPACE_ENV: ray_namespace()},
            },
        },
    ).bind(
        model_id=model_id,
        model_name=model_name,
        gpu_deployment_names=gpu_deployment_names,
        replica_metadata=replica_metadata,
        chat_template_kwargs=chat_template_kwargs or {},
        idle_timeout=idle_timeout,
        sleep_timeout=sleep_timeout,
        pooling=pooling,
        multimodal=multimodal,
        chat_template=chat_template,
    )
    try:
        serve.run(router, name=model_id, route_prefix=f"/{model_id}")
    except Exception as e:
        apps = serve.status().applications or {}
        app = apps.get(model_id)
        details = []
        if app is not None:
            for dep_name, dep in (app.deployments or {}).items():
                details.append(f"{dep_name}: {dep.status} — {dep.message}")
            serve.delete(name=model_id)
        if details:
            raise DeployError("; ".join(details)) from e
        raise
    return True


@ray.remote(num_cpus=0)
class DeployService:
    """Singleton actor — serializes model deploys across apps."""

    def __init__(self):
        self._busy = False

    def code_rev(self) -> int:
        """Working-dir generation; client recreates the actor on mismatch."""
        return HIVE_DEPLOY_CODE_REV

    def is_busy(self) -> bool:
        """True while deploy_models is in flight (blocks code_rev kill)."""
        return self._busy

    def deploy_models(self, model_configs: dict, vllm_kwargs: dict | None = None) -> dict:
        """Deploy one or more models; returns per-replica plan dicts."""
        self._busy = True
        try:
            serve.start(
                proxy_location="HeadOnly",
                http_options={"host": "0.0.0.0", "port": 8000},
            )
            vllm_kwargs = vllm_kwargs or {}
            results = {}
            for model_id, config in model_configs.items():
                results[model_id] = self._deploy_model(
                    model_id, config, vllm_kwargs.get(model_id, {})
                )
            return results
        finally:
            self._busy = False


    def _deploy_model(self, model_id: str, config: dict, model_vllm_kwargs: dict) -> dict:
        """Run the full deploy pipeline for one model."""
        if "max_input_prompt_length" not in config or "max_output_prompt_length" not in config:
            raise ConfigError("max_input_prompt_length and max_output_prompt_length are required")
        from .ray_utils.placement import validate_auto_input_config
        validate_auto_input_config(config)

        # Serve-frontend only — not valid on LLM()/EngineArgs; applied by ModelRouter.
        model_vllm_kwargs = dict(model_vllm_kwargs)
        chat_template_kwargs = model_vllm_kwargs.pop("default_chat_template_kwargs", None) or {}
        chat_template = model_vllm_kwargs.pop("chat_template", None)
        model_vllm_kwargs.pop("tensor_parallel_size", None)
        model_vllm_kwargs.pop("distributed_executor_backend", None)

        registry = get_gpu_registry()
        assert_model_id_free(model_id, registry)

        # Pin to any advertised hive GPU custom-resource (workers have transformers;
        # head often does not). Avoid num_gpus=0.01 — fights lexis / fractional GPU.
        gpu_res = next(
            (k for pool in (ray.available_resources(), ray.cluster_resources())
             for k, v in pool.items() if "_gpu" in k and v >= 0.01),
            None,
        )
        fetch_opts = {"num_cpus": 0}
        if gpu_res:
            fetch_opts["resources"] = {gpu_res: 0.01}
        else:
            fetch_opts["num_gpus"] = 0.01
        hf_params = normalize_hf_config(
            ray.get(fetch_hf_config_dict.options(**fetch_opts).remote(config["name"]))
        )
        gpu_map = ray.get(registry.get_all_gpus.remote())

        from ray_hive.core.model_specs.factory import is_multimodal_hf, is_pooling_kwargs
        pooling = is_pooling_kwargs(model_vllm_kwargs)
        multimodal = is_multimodal_hf(hf_params) and not model_vllm_kwargs.get(
            "language_model_only"
        )

        planned = plan_replica_groups(gpu_map, config, hf_params, model_vllm_kwargs, model_id)

        replica_jobs = []
        gpu_mapping = {}
        replica_metadata = {}
        reserved_ids: list[str] = []

        try:
            for replica_id, entry in planned.items():
                plan = entry["plan"]
                group = entry["group"]
                gpu_keys = entry["gpu_keys"]
                tp_size = entry["tp_size"]
                max_model_len = entry["max_model_len"]

                for gpu_info in group:
                    if gpu_info["available_gb"] < plan["total_vram_gb"]:
                        raise InsufficientVramError(
                            f"GPU {gpu_info['gpu_key']} has {gpu_info['available_gb']:.2f}GB available, "
                            f"need {plan['total_vram_gb']:.2f}GB for TP plan",
                            gpu=gpu_info["gpu_key"],
                            available_gb=gpu_info["available_gb"],
                            need_gb=plan["total_vram_gb"],
                        )
                gpu_vram = {g["gpu_key"]: plan["total_vram_gb"] for g in group}
                if not ray.get(registry.reserve_replica_group.remote(replica_id, gpu_vram)):
                    raise InsufficientVramError(
                        f"Could not reserve VRAM for {replica_id} on {list(gpu_vram)} "
                        f"(need {plan['total_vram_gb']:.2f}GB each) — concurrent deploy or stale plan",
                        gpu=next(iter(gpu_vram)),
                        available_gb=None,
                        need_gb=plan["total_vram_gb"],
                    )
                reserved_ids.append(replica_id)

                engine_kwargs = {
                    "model": config["name"],
                    "max_model_len": max_model_len,
                    "max_num_seqs": plan["max_num_seqs"],
                    "max_num_batched_tokens": plan["max_num_batched_tokens"],
                    "gpu_memory_utilization": plan.get(
                        "vllm_gpu_memory_utilization", plan["gpu_memory_utilization"]
                    ),
                    "enforce_eager": False,
                }
                # Caller may set kv_cache_memory_bytes itself; that value wins.
                if "kv_cache_memory_bytes" not in model_vllm_kwargs:
                    engine_kwargs["kv_cache_memory_bytes"] = int(plan["kv_cache_bytes"])
                engine_kwargs.update(model_vllm_kwargs)
                if config.get("sleep_timeout", -1) > 0:
                    engine_kwargs["enable_sleep_mode"] = True
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
                    "pooling": pooling,
                    "multimodal": multimodal,
                })
                gpu_mapping[replica_id] = gpu_keys
                replica_metadata[replica_id] = plan
        except Exception:
            if reserved_ids:
                ray.get(registry.clear_replicas.remote(reserved_ids))
            raise

        # Clamp util against live available so embed-after-name does not OOM.
        for job in replica_jobs:
            ek = job["engine_kwargs"]
            util = float(ek.get("gpu_memory_utilization") or 0.9)
            for gkey in gpu_mapping[job["replica_id"]]:
                view = gpu_map.get(gkey) or {}
                total = float(view.get("total") or 0)
                avail = float(view.get("available") or 0)
                if total > 0 and avail < total * util:
                    ek["gpu_memory_utilization"] = max(0.05, min(util, avail / total * 0.98))

        deploy_futures = [
            deploy_single.options(
                resources={name: 0.01 for name in job["resource_names"]}, num_cpus=0
            ).remote(
                job["replica_id"],
                job["model_id"],
                job["target_gpu_id"],
                job["engine_kwargs"],
                job["resource_names"],
                f"/{job['replica_id']}",
                job["pooling"],
                job["multimodal"],
            )
            for job in replica_jobs
        ]
        ok_jobs = []
        fail_details = []
        failed_ids = []
        for job, fut in zip(replica_jobs, deploy_futures):
            try:
                ray.get(fut)
                ok_jobs.append(job)
            except Exception as e:
                failed_ids.append(job["replica_id"])
                apps = serve.status().applications or {}
                app = apps.get(job["replica_id"])
                if app is not None:
                    for dep_name, dep in (app.deployments or {}).items():
                        fail_details.append(f"{dep_name}: {dep.status} — {dep.message}")
                    serve.delete(name=job["replica_id"])
                else:
                    fail_details.append(
                        f"{job['replica_id']}: {type(e).__name__}: {e} "
                        "(check worker nvidia-smi / pod logs for CUDA OOM)"
                    )
        if failed_ids:
            ray.get(registry.clear_replicas.remote(failed_ids))
        want_all = config.get("replicas", 1) != -1
        if not ok_jobs or (want_all and failed_ids):
            for job in ok_jobs:
                apps = serve.status().applications or {}
                if job["replica_id"] in apps:
                    serve.delete(name=job["replica_id"])
            ray.get(registry.clear_replicas.remote(
                [j["replica_id"] for j in ok_jobs]
            ))
            msg = "; ".join(fail_details) if fail_details else "replica deploy failed"
            raise DeployError(msg)
        replica_jobs = ok_jobs
        gpu_mapping = {j["replica_id"]: gpu_mapping[j["replica_id"]] for j in replica_jobs}
        replica_metadata = {
            j["replica_id"]: replica_metadata[j["replica_id"]] for j in replica_jobs
        }

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
        ray.get(deploy_router.options(resources={router_resource: 0.01}, num_cpus=0).remote(
            model_id,
            config["name"],
            gpu_deployment_names,
            replica_metadata,
            router_resource,
            chat_template_kwargs,
            config.get("idle_timeout", -1),
            config.get("sleep_timeout", -1),
            pooling,
            multimodal,
            chat_template,
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
    ns = ray_namespace()
    try:
        actor = ray.get_actor("deploy_service", namespace=ns)
        try:
            if ray.get(actor.code_rev.remote()) == HIVE_DEPLOY_CODE_REV:
                return actor
        except Exception:
            pass
        # Never kill mid-deploy — wait for drain, then recreate.
        for _ in range(120):
            try:
                if not ray.get(actor.is_busy.remote()):
                    break
            except Exception:
                break
            time.sleep(1)
        ray.kill(actor)
        time.sleep(1)
    except ValueError:
        pass
    last = None
    for attempt in range(16):
        try:
            # Ray Client can return InProgressSentinel while working_dir uploads;
            # brief backoff then retry actor creation.
            if attempt:
                time.sleep(min(2 * attempt, 10))
            # Do not set working_dir here: Ray Jobs forbid path URIs on actors.
            # Inherit the creator job/client working_dir; bump HIVE_DEPLOY_CODE_REV
            # to force recreate when ModelRouter/deploy code changes.
            return DeployService.options(
                name="deploy_service",
                namespace=ns,
                lifetime="detached",
                max_concurrency=1,
                runtime_env={"env_vars": {NAMESPACE_ENV: ns}},
            ).remote()
        except Exception as e:
            last = e
            if "InProgressSentinel" not in type(e).__name__ and "InProgressSentinel" not in str(e):
                # still retry a few times for name-conflict races after kill
                if attempt >= 3 and "already exists" not in str(e).lower():
                    time.sleep(0.5)
    raise last
