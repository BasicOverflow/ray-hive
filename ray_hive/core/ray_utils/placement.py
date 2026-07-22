"""GPU placement helpers used by deploy planning."""
from ray_hive.core.model_specs.planner import build_vram_reqs

from .naming import gpu_info_entry


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


def resolve_pinned_gpu(gpu_map: dict, gpu_key: str, min_vram_gb: float) -> dict:
    """Resolve an explicit GPU pin from the full registry (not the filtered eligible list)."""
    if gpu_key not in gpu_map:
        raise ValueError(f"GPU {gpu_key} not in registry. Known: {sorted(gpu_map)}")
    gpu_info = gpu_map[gpu_key]
    if gpu_info["available"] < min_vram_gb:
        raise ValueError(
            f"GPU {gpu_key} has {gpu_info['available']:.2f}GB available, "
            f"need {min_vram_gb:.2f}GB (weights+overhead+misc headroom)"
        )
    return gpu_info_entry(gpu_key, gpu_info)


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


def resolve_cpu_ram_budget(cfg: float, available_gb: float) -> float:
    """Resolve host GiB budget: 0 off, -1 = 85% free VRAM, >0 hard GiB."""
    if cfg < -1:
        raise ValueError(f"cpu_ram_per_instance must be >= -1, got {cfg}")
    if cfg == -1:
        return 0.85 * float(available_gb)
    if cfg < 0:
        raise ValueError(f"cpu_ram_per_instance must be 0, -1, or >0, got {cfg}")
    return float(cfg)


def resolve_cpu_spill(
    cpu_ram_gb: float,
    weight_need_gb: float,
    gpu_budget_gb: float,
    tp_size: int,
) -> tuple[float, float]:
    """
    Split host budget: weights on GPU if they fit; else spill overflow; remainder → KV.

    Returns (cpu_offload_gb per rank, kv_offload_gb engine-total).
    """
    if cpu_ram_gb <= 0:
        return 0.0, 0.0
    tp_size = max(1, int(tp_size))
    if weight_need_gb <= gpu_budget_gb:
        return 0.0, float(cpu_ram_gb)
    cpu_offload = weight_need_gb - gpu_budget_gb
    host_for_weights = cpu_offload * tp_size
    if host_for_weights > cpu_ram_gb:
        raise ValueError(
            f"Weight spill needs {host_for_weights:.2f}GB host RAM "
            f"({cpu_offload:.2f}GB/rank x TP={tp_size}), "
            f"but cpu_ram_per_instance budget is {cpu_ram_gb:.2f}GB"
        )
    return float(cpu_offload), float(cpu_ram_gb - host_for_weights)
