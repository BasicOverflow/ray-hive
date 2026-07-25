"""GPU placement helpers used by deploy planning."""
from ray_hive.core.model_specs.planner import build_vram_reqs

from .hardware import count_by_host


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
    """TP needs headroom for NCCL / cudagraph peaks; packing ~0.97 OOMs on small cards."""
    return 0.80 if tp_size > 1 else 0.97


def replicas_per_host(gpu_groups: list[list[dict]]) -> dict[str, int]:
    """Count replica/TP groups landing on each host."""
    return count_by_host(group[0]["gpu_key"] for group in gpu_groups)
