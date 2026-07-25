"""GPU placement helpers used by deploy planning."""
from ray_hive.core.gpu_alloc import assert_cpu_ram_tp_allowed, resolve_group_cpu_spill
from ray_hive.core.gpu_alloc.cpu_spill import TP1_BUDGET_FRAC
from ray_hive.core.model_specs.planner import build_vram_reqs, plan_deployment

from .hardware import count_by_host, host_memory_available_gb
from .naming import deployment_name


def fixed_non_kv_gb(vram_reqs) -> float:
    """Minimum per-GPU VRAM to load weights + overhead (before KV)."""
    return (
        vram_reqs.calc_system_overhead_gb()
        + vram_reqs.calc_weights_gb()
        + vram_reqs.calc_misc_vram_gb()
        + 0.25
    )


def build_vram_reqs_for_tp(hf_params, attention_cls, model_vllm_kwargs, tp_size: int):
    """Build VramReqs for a given TP size."""
    return build_vram_reqs(
        hf_params,
        attention_cls=attention_cls,
        tensor_parallel_size=tp_size,
        **model_vllm_kwargs,
    )


def chunk_gpu_groups(gpus: list[dict], tp_size: int) -> list[list[dict]]:
    """Split a flat GPU list into contiguous TP groups of size tp_size."""
    if tp_size == 1:
        return [[g] for g in gpus]
    if len(gpus) % tp_size != 0:
        raise ValueError(
            f"Got {len(gpus)} GPUs but tensor_parallel_size={tp_size} "
            f"(need a multiple of {tp_size})"
        )
    return [gpus[i : i + tp_size] for i in range(0, len(gpus), tp_size)]


def gpu_budget_frac(tp_size: int) -> float:
    """TP needs more headroom for NCCL / cudagraph peaks than TP=1."""
    return 0.80 if tp_size > 1 else TP1_BUDGET_FRAC


def replicas_per_host(gpu_groups: list[list[dict]]) -> dict[str, int]:
    """Count replica/TP groups landing on each host."""
    return count_by_host(group[0]["gpu_key"] for group in gpu_groups)


def plan_replica_groups(
    gpu_map: dict,
    config: dict,
    hf_params: dict,
    model_vllm_kwargs: dict,
    model_id: str = "estimate",
) -> dict:
    """
    Dry-run the same packing deploy uses. Returns
    {replica_id: {plan, gpu_keys, group, cpu_offload, kv_offload, tp_size, max_model_len}}.
    """
    from .select_gpus import resolve_target_gpus

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
    weight_need = fixed_non_kv_gb(vram_reqs)
    results = {}

    for group in gpu_groups:
        gpu_keys = [g["gpu_key"] for g in group]
        bottleneck = min(group, key=lambda g: g["available_gb"])
        avail = min(g["available_gb"] for g in group)
        host = group[0]["gpu_key"].split(":")[0]
        if tp_size > 1:
            per_gpu_budget = avail * gpu_budget_frac(tp_size)
            cpu_ram = cpu_offload = kv_offload = 0.0
        else:
            per_gpu_budget, cpu_ram, cpu_offload, kv_offload, _ = resolve_group_cpu_spill(
                weight_need, avail, cpu_ram_cfg, host_memory_available_gb(host), per_host[host]
            )
        plan = plan_deployment(
            vram_reqs,
            vram_budget_gb=per_gpu_budget,
            live_total_vram_gb=bottleneck["total_gb"],
            max_model_len=max_model_len,
            input_len=config["max_input_prompt_length"],
            output_len=config["max_output_prompt_length"],
            max_num_batched_tokens_override=config.get("max_num_batched_tokens"),
            max_num_seqs_override=config.get("max_num_seqs"),
            cpu_kv_offload_gb=kv_offload,
            cpu_weight_offload_gb=cpu_offload,
            live_available_vram_gb=avail,
        )
        plan["tensor_parallel_size"] = tp_size
        plan["weights_gb"] = vram_reqs.calc_weights_gb() * tp_size
        plan["weight_need_gb"] = weight_need
        plan["cpu_ram_budget_gb"] = cpu_ram

        replica_id = deployment_name(model_id, gpu_keys)
        results[replica_id] = {
            "plan": plan,
            "gpu_keys": gpu_keys,
            "group": group,
            "cpu_offload": cpu_offload,
            "kv_offload": kv_offload,
            "tp_size": tp_size,
            "max_model_len": max_model_len,
        }
    return results
