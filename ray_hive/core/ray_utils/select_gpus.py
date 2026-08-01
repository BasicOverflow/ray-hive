"""Resolve target GPUs + TP size for a deploy (pins and auto allocation)."""
from ray_hive.core.gpu_alloc import required_min_compute_cap
from ray_hive.core.ray_gpu_alloc import RayPerformanceAllocator, RayTensorParallelAllocator
from ray_hive.errors import (
    ArchRequirementError,
    GpuNotFoundError,
    InsufficientVramError,
    InvalidGpuPinError,
    NoPlacementError,
    PlacementError,
)

from .hardware import compute_cap, gpu_inventory_lines, max_gpus_on_any_host
from .naming import gpu_info_entry
from .placement import build_vram_reqs_for_tp, fixed_non_kv_gb
from .tensor_parallel import assert_tp_shardable, tp_shardable


def _resolve_pinned_gpu(
    gpu_map: dict,
    gpu_key: str,
    min_vram_gb: float,
    hf_params: dict,
    vllm_kwargs: dict,
) -> dict:
    if gpu_key not in gpu_map:
        raise GpuNotFoundError(f"GPU {gpu_key} not in registry. Known: {sorted(gpu_map)}")
    gpu_info = gpu_map[gpu_key]
    if gpu_info["available"] < min_vram_gb:
        raise InsufficientVramError(
            f"GPU {gpu_key} has {gpu_info['available']:.2f}GB available, "
            f"need {min_vram_gb:.2f}GB (weights+overhead+misc headroom)",
            gpu=gpu_key,
            available_gb=gpu_info["available"],
            need_gb=min_vram_gb,
        )
    min_cap = required_min_compute_cap(hf_params, vllm_kwargs)
    if min_cap is not None and compute_cap(gpu_info) < min_cap:
        raise ArchRequirementError(
            f"GPU {gpu_key} is compute_cap={compute_cap(gpu_info)}, "
            f"but this model needs >={min_cap} (native FP8)"
        )
    return gpu_info_entry(gpu_key, gpu_info)


def resolve_target_gpus(
    gpu_map: dict,
    replicas: int,
    gpu,
    hf_params: dict,
    allocation_cls,
    attention_cls,
    model_vllm_kwargs: dict,
    sleep_mode: bool = False,
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
            return _pin_single(
                gpu_map, gpu, hf_params, attention_cls, model_vllm_kwargs, sleep_mode,
            )

        if isinstance(gpu, list):
            if not gpu:
                raise InvalidGpuPinError("gpu=[] is empty")
            if len(gpu) == 1:
                return _pin_single(
                    gpu_map, gpu[0], hf_params, attention_cls, model_vllm_kwargs, sleep_mode,
                )
            if replicas == len(gpu) and replicas > 1:
                return _pin_multi_tp1(
                    gpu_map, gpu, hf_params, attention_cls, model_vllm_kwargs, sleep_mode,
                )
            if replicas != 1:
                raise InvalidGpuPinError(
                    "Pinned gpu=[...] with len>1 is either one TP group (replicas=1) "
                    "or N single-GPU pins (replicas=len(gpu)). One deploy creates at most "
                    "one TP replica — for a second TP copy, call deploy_model again with a "
                    f"different model_id / pin. Got replicas={replicas}, len(gpu)={len(gpu)}."
                )
            return _pin_tp_group(
                gpu_map, gpu, hf_params, attention_cls, model_vllm_kwargs, sleep_mode,
            )

        assert False, "gpu must be a string, list of strings, or None"

    return _auto_place(
        gpu_map, replicas, hf_params, allocation_cls, attention_cls, model_vllm_kwargs,
        sleep_mode,
    )


def _pin_single(gpu_map, gpu_key, hf_params, attention_cls, model_vllm_kwargs, sleep_mode=False):
    tp_size = 1
    vram_reqs = build_vram_reqs_for_tp(hf_params, attention_cls, model_vllm_kwargs, tp_size)
    weight_need = fixed_non_kv_gb(vram_reqs, sleep_mode=sleep_mode)
    return (
        tp_size,
        [_resolve_pinned_gpu(gpu_map, gpu_key, weight_need, hf_params, model_vllm_kwargs)],
        vram_reqs,
    )


def _pin_multi_tp1(gpu_map, gpu_keys, hf_params, attention_cls, model_vllm_kwargs, sleep_mode=False):
    tp_size = 1
    vram_reqs = build_vram_reqs_for_tp(hf_params, attention_cls, model_vllm_kwargs, tp_size)
    weight_need = fixed_non_kv_gb(vram_reqs, sleep_mode=sleep_mode)
    return (
        tp_size,
        [
            _resolve_pinned_gpu(gpu_map, g, weight_need, hf_params, model_vllm_kwargs)
            for g in gpu_keys
        ],
        vram_reqs,
    )


def _pin_tp_group(gpu_map, gpu_keys, hf_params, attention_cls, model_vllm_kwargs, sleep_mode=False):
    tp_size = len(gpu_keys)
    hosts = {g.split(":")[0] for g in gpu_keys}
    if len(hosts) != 1:
        raise PlacementError(f"Same-node TP only — pinned GPUs span hosts {sorted(hosts)}")
    assert_tp_shardable(hf_params, tp_size)
    vram_reqs = build_vram_reqs_for_tp(hf_params, attention_cls, model_vllm_kwargs, tp_size)
    weight_need = fixed_non_kv_gb(vram_reqs, sleep_mode=sleep_mode)
    return (
        tp_size,
        [
            _resolve_pinned_gpu(gpu_map, g, weight_need, hf_params, model_vllm_kwargs)
            for g in gpu_keys
        ],
        vram_reqs,
    )


def _auto_place(
    gpu_map, replicas, hf_params, allocation_cls, attention_cls, model_vllm_kwargs,
    sleep_mode=False,
):
    if not gpu_map:
        raise PlacementError("No GPUs in registry — cannot place model")

    vram_reqs_1 = build_vram_reqs_for_tp(hf_params, attention_cls, model_vllm_kwargs, 1)
    weight_need_1 = fixed_non_kv_gb(vram_reqs_1, sleep_mode=sleep_mode)
    single_cls = allocation_cls or RayPerformanceAllocator
    chosen = single_cls().select(
        gpu_map, replicas, weight_need_1, hf_params, model_vllm_kwargs
    )
    if chosen:
        if replicas != -1 and len(chosen) < replicas:
            raise PlacementError(
                f"Only {len(chosen)} GPU(s) can hold the model "
                f"(need {weight_need_1:.2f}GB each), "
                f"but replicas={replicas}.\nCluster:\n{gpu_inventory_lines(gpu_map)}"
            )
        return 1, [gpu_info_entry(k, g) for k, g in chosen], vram_reqs_1

    max_tp = max_gpus_on_any_host(gpu_map)
    best_partial = None
    for tp_size in range(2, max_tp + 1):
        if not tp_shardable(hf_params, tp_size):
            continue
        vram_reqs = build_vram_reqs_for_tp(hf_params, attention_cls, model_vllm_kwargs, tp_size)
        weight_need = fixed_non_kv_gb(vram_reqs, sleep_mode=sleep_mode)
        alloc_kwargs = dict(model_vllm_kwargs)
        alloc_kwargs["tensor_parallel_size"] = tp_size
        chosen = RayTensorParallelAllocator().select(
            gpu_map, replicas, weight_need, hf_params, alloc_kwargs
        )
        n_groups = len(chosen) // tp_size if chosen else 0
        if n_groups == 0:
            continue
        if replicas == -1 or n_groups >= replicas:
            return tp_size, [gpu_info_entry(k, g) for k, g in chosen], vram_reqs
        best_partial = (tp_size, n_groups, weight_need)

    if best_partial is not None:
        tp_size, n_groups, min_vram = best_partial
        raise PlacementError(
            f"Only {n_groups} same-node TP={tp_size} group(s) fit "
            f"(need {min_vram:.2f}GB/GPU on GPU VRAM), but replicas={replicas}.\n"
            f"Cluster:\n{gpu_inventory_lines(gpu_map)}"
        )

    raise NoPlacementError(
        f"No GPU set in the cluster can support this model "
        f"(need >={weight_need_1:.2f}GB on one GPU, or same-node TP=2..{max_tp} on GPU VRAM only).\n"
        f"Cluster:\n{gpu_inventory_lines(gpu_map)}"
    )
