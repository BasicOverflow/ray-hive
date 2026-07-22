"""
Deployment service — singleton Ray actor that serializes deploy/shutdown.

DeployService runs the full per-model pipeline: HF config → planner → GPU
selection → replica reservation → parallel RayLLMActor deploy → ModelRouter.
"""
import time

import ray
from ray import serve

from .gpu_registry import get_gpu_registry
from .model_specs.planner import normalize_hf_config, plan_deployment
from .ray_gpu_alloc import RayPerformanceAllocator, RayTensorParallelAllocator
from .ray_utils import (
    assert_model_id_free,
    assert_tp_shardable,
    build_vram_reqs_for_tp,
    chunk_gpu_groups,
    deployment_name,
    fixed_non_kv_gb,
    gpu_info_entry,
    gpu_inventory_lines,
    max_gpus_on_any_host,
    resolve_cpu_ram_budget,
    resolve_cpu_spill,
    resolve_pinned_gpu,
    tp_shardable,
)


def _budget_frac(tp_size: int) -> float:
    """TP needs headroom for NCCL / cudagraph peaks; packing ~0.97 OOMs on small cards."""
    return 0.80 if tp_size > 1 else 0.97


def _pin_weight_need_gb(weight_need: float, available_gb: float, cpu_ram_cfg: float, tp_size: int) -> float:
    """On-GPU GiB required after optional host weight spill; raises if spill budget too small."""
    gpu_budget = available_gb * _budget_frac(tp_size)
    cpu_ram = resolve_cpu_ram_budget(cpu_ram_cfg, available_gb)
    cpu_offload, _ = resolve_cpu_spill(cpu_ram, weight_need, gpu_budget, tp_size)
    return weight_need - cpu_offload


def _resolve_target_gpus(
    gpu_map: dict,
    replicas: int,
    gpu,
    hf_params: dict,
    allocation_cls,
    attention_cls,
    model_vllm_kwargs: dict,
    cpu_ram_cfg: float = 0,
) -> tuple[int, list[dict], object]:
    """
    Resolve placement and TP size.

    Returns (tp_size, flat gpu info list, vram_reqs for that tp_size).

    - gpu=str / gpu=[one]: TP=1 pin
    - gpu=[a,b,...] + replicas == len(list) + replicas > 1: N single-GPU pins
    - gpu=[a,b,...] + replicas == 1: one same-node TP group
    - gpu=None: try TP=1; if nothing fits, escalate same-node TP=2,3,...
    """
    if gpu is not None:
        if isinstance(gpu, str):
            tp_size = 1
            vram_reqs = build_vram_reqs_for_tp(hf_params, attention_cls, model_vllm_kwargs, tp_size)
            weight_need = fixed_non_kv_gb(vram_reqs)
            if gpu not in gpu_map:
                raise ValueError(f"GPU {gpu} not in registry. Known: {sorted(gpu_map)}")
            avail = float(gpu_map[gpu]["available"])
            need = _pin_weight_need_gb(weight_need, avail, cpu_ram_cfg, tp_size)
            return tp_size, [resolve_pinned_gpu(gpu_map, gpu, need)], vram_reqs

        if isinstance(gpu, list):
            if not gpu:
                raise ValueError("gpu=[] is empty")
            if len(gpu) == 1:
                tp_size = 1
                vram_reqs = build_vram_reqs_for_tp(hf_params, attention_cls, model_vllm_kwargs, tp_size)
                weight_need = fixed_non_kv_gb(vram_reqs)
                if gpu[0] not in gpu_map:
                    raise ValueError(f"GPU {gpu[0]} not in registry. Known: {sorted(gpu_map)}")
                avail = float(gpu_map[gpu[0]]["available"])
                need = _pin_weight_need_gb(weight_need, avail, cpu_ram_cfg, tp_size)
                return tp_size, [resolve_pinned_gpu(gpu_map, gpu[0], need)], vram_reqs

            if replicas == len(gpu) and replicas > 1:
                tp_size = 1
                vram_reqs = build_vram_reqs_for_tp(hf_params, attention_cls, model_vllm_kwargs, tp_size)
                weight_need = fixed_non_kv_gb(vram_reqs)
                resolved = []
                for g in gpu:
                    if g not in gpu_map:
                        raise ValueError(f"GPU {g} not in registry. Known: {sorted(gpu_map)}")
                    avail = float(gpu_map[g]["available"])
                    need = _pin_weight_need_gb(weight_need, avail, cpu_ram_cfg, tp_size)
                    resolved.append(resolve_pinned_gpu(gpu_map, g, need))
                return tp_size, resolved, vram_reqs

            if replicas != 1:
                raise ValueError(
                    "Pinned gpu=[...] with len>1 is either one TP group (replicas=1) "
                    "or N single-GPU pins (replicas=len(gpu)). One deploy creates at most "
                    "one TP replica — for a second TP copy, call deploy_model again with a "
                    f"different model_id / pin. Got replicas={replicas}, len(gpu)={len(gpu)}."
                )

            tp_size = len(gpu)
            hosts = {g.split(":")[0] for g in gpu}
            if len(hosts) != 1:
                raise ValueError(
                    f"Same-node TP only — pinned GPUs span hosts {sorted(hosts)}"
                )
            assert_tp_shardable(hf_params, tp_size)
            vram_reqs = build_vram_reqs_for_tp(hf_params, attention_cls, model_vllm_kwargs, tp_size)
            weight_need = fixed_non_kv_gb(vram_reqs)
            resolved = []
            short = []
            for g in gpu:
                if g not in gpu_map:
                    raise ValueError(f"GPU {g} not in registry. Known: {sorted(gpu_map)}")
                info = gpu_map[g]
                resolved.append(gpu_info_entry(g, info))
            avail = min(e["available_gb"] for e in resolved)
            try:
                need = _pin_weight_need_gb(weight_need, avail, cpu_ram_cfg, tp_size)
            except ValueError as e:
                raise ValueError(
                    f"Model does not fit on pinned gpu={gpu} (TP={tp_size}): {e}"
                ) from e
            for entry in resolved:
                if entry["available_gb"] < need:
                    short.append(
                        f"{entry['gpu_key']}: {entry['available_gb']:.2f}GB avail "
                        f"(need {need:.2f}GB/GPU after TP={tp_size})"
                    )
            if short:
                raise ValueError(
                    f"Model does not fit on pinned gpu={gpu} (TP={tp_size}): "
                    + "; ".join(short)
                )
            return tp_size, resolved, vram_reqs

        assert False, "gpu must be a string, list of strings, or None"

    if not gpu_map:
        raise ValueError("No GPUs in registry — cannot place model")

    # Auto: prefer single-GPU, then escalate TP on same-node packs.
    vram_reqs_1 = build_vram_reqs_for_tp(hf_params, attention_cls, model_vllm_kwargs, 1)
    weight_need_1 = fixed_non_kv_gb(vram_reqs_1)
    # Hard cpu_ram credits selection floor; -1 is per-GPU so keep weight_need for filter.
    if cpu_ram_cfg > 0:
        min_1 = max(0.01, weight_need_1 - float(cpu_ram_cfg))
    else:
        min_1 = weight_need_1
    single_cls = allocation_cls or RayPerformanceAllocator
    chosen = single_cls().select(gpu_map, replicas, min_1, hf_params, model_vllm_kwargs)
    if chosen:
        if replicas != -1 and len(chosen) < replicas:
            raise ValueError(
                f"Only {len(chosen)} GPU(s) can hold the model (need {min_1:.2f}GB each), "
                f"but replicas={replicas}.\nCluster:\n{gpu_inventory_lines(gpu_map)}"
            )
        return 1, [gpu_info_entry(k, g) for k, g in chosen], vram_reqs_1

    max_tp = max_gpus_on_any_host(gpu_map)
    best_partial = None  # (tp_size, n_groups, min_vram) when some groups fit but < replicas
    for tp_size in range(2, max_tp + 1):
        if not tp_shardable(hf_params, tp_size):
            continue
        vram_reqs = build_vram_reqs_for_tp(hf_params, attention_cls, model_vllm_kwargs, tp_size)
        weight_need = fixed_non_kv_gb(vram_reqs)
        if cpu_ram_cfg > 0:
            min_vram = max(0.01, weight_need - float(cpu_ram_cfg) / tp_size)
        else:
            min_vram = weight_need
        alloc_kwargs = dict(model_vllm_kwargs)
        alloc_kwargs["tensor_parallel_size"] = tp_size
        # TP packing always uses the TP allocator (symmetric same-node sets).
        chosen = RayTensorParallelAllocator().select(
            gpu_map, replicas, min_vram, hf_params, alloc_kwargs
        )
        n_groups = len(chosen) // tp_size if chosen else 0
        if n_groups == 0:
            continue
        if replicas == -1 or n_groups >= replicas:
            return tp_size, [gpu_info_entry(k, g) for k, g in chosen], vram_reqs
        best_partial = (tp_size, n_groups, min_vram)

    if best_partial is not None:
        tp_size, n_groups, min_vram = best_partial
        raise ValueError(
            f"Only {n_groups} same-node TP={tp_size} group(s) fit "
            f"(need {min_vram:.2f}GB/GPU), but replicas={replicas}.\n"
            f"Cluster:\n{gpu_inventory_lines(gpu_map)}"
        )

    raise ValueError(
        f"No GPU set in the cluster can support this model "
        f"(need >={min_1:.2f}GB on one GPU, or same-node TP=2..{max_tp}).\n"
        f"Cluster:\n{gpu_inventory_lines(gpu_map)}"
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
    gpu_fraction: float,
    resource_names: list[str],
    route_prefix: str,
    memory_bytes: int,
):
    """Deploy one RayLLMActor replica on a GPU worker (imports vllm here only)."""
    from ray_hive.core.ray_llm_actor import RayLLMActor

    resources = {name: 0.01 for name in resource_names}
    # Ray remaps CUDA_VISIBLE_DEVICES when num_gpus>0; that breaks multi-GPU TP.
    # Pin devices via runtime_env + custom resources; keep fractional num_gpus only for single-GPU.
    multi_gpu = len(resource_names) > 1
    env_vars = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": target_gpu_id,
    }
    if multi_gpu:
        env_vars["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
    deployment = RayLLMActor.options(
        name=replica_id,
        ray_actor_options={
            "num_gpus": 0 if multi_gpu else gpu_fraction,
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
        registry = get_gpu_registry()
        deployment = ray.get(registry.get_deployment.remote(model_id))
        replica_ids = list(deployment["replicas"].keys()) if deployment else []

        apps = serve.status().applications or {}
        for app_name in (model_id, *replica_ids):
            if app_name in apps:
                serve.delete(name=app_name)

        if replica_ids:
            ray.get(registry.clear_replicas.remote(replica_ids))
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
        tp_size, target_gpus, vram_reqs = _resolve_target_gpus(
            gpu_map,
            config.get("replicas", -1),
            config.get("gpu"),
            hf_params,
            config.get("allocation_cls"),
            config.get("attention_cls"),
            model_vllm_kwargs,
            cpu_ram_cfg=cpu_ram_cfg,
        )
        gpu_groups = chunk_gpu_groups(target_gpus, tp_size)

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
            per_gpu_budget = avail * _budget_frac(tp_size)
            cpu_ram = resolve_cpu_ram_budget(cpu_ram_cfg, avail)
            cpu_offload, kv_offload = resolve_cpu_spill(
                cpu_ram, weight_need, per_gpu_budget, tp_size
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
            # Match single-GPU pinning: small Ray num_gpus + custom resources; CUDA_VISIBLE_DEVICES selects devices.
            per_card = max(0.01, round(plan["total_vram_gb"] / bottleneck["total_gb"], 2))
            gpu_fraction = per_card * tp_size
            memory_bytes = int((2 + kv_offload + cpu_offload * tp_size) * 1024 ** 3)
            replica_jobs.append({
                "replica_id": replica_id,
                "model_id": model_id,
                "target_gpu_id": ",".join(g["gpu_id"] for g in group),
                "engine_kwargs": engine_kwargs,
                "gpu_fraction": gpu_fraction,
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
            future = deploy_single.options(resources=resources).remote(
                job["replica_id"],
                job["model_id"],
                job["target_gpu_id"],
                job["engine_kwargs"],
                job["gpu_fraction"],
                job["resource_names"],
                f"/{job['replica_id']}",
                job["memory_bytes"],
            )
            deploy_futures.append(future)
        ray.get(deploy_futures)

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
