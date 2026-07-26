"""Resolve target GPUs + TP size for a deploy (pins and auto allocation)."""
from ray_hive.core.gpu_alloc import on_gpu_weight_need_gb
from ray_hive.core.ray_gpu_alloc import RayPerformanceAllocator, RayTensorParallelAllocator

from .hardware import count_by_host, gpu_inventory_lines, host_memory_available_gb, max_gpus_on_any_host
from .naming import gpu_info_entry
from .placement import build_vram_reqs_for_tp, fixed_non_kv_gb
from .tensor_parallel import assert_tp_shardable, tp_shardable


def _resolve_pinned_gpu(gpu_map: dict, gpu_key: str, min_vram_gb: float) -> dict:
    if gpu_key not in gpu_map:
        raise ValueError(f"GPU {gpu_key} not in registry. Known: {sorted(gpu_map)}")
    gpu_info = gpu_map[gpu_key]
    if gpu_info["available"] < min_vram_gb:
        raise ValueError(
            f"GPU {gpu_key} has {gpu_info['available']:.2f}GB available, "
            f"need {min_vram_gb:.2f}GB (weights+overhead+misc headroom)"
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
    cpu_ram_cfg: float = 0,
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
                gpu_map, gpu, hf_params, attention_cls, model_vllm_kwargs, cpu_ram_cfg, sleep_mode
            )

        if isinstance(gpu, list):
            if not gpu:
                raise ValueError("gpu=[] is empty")
            if len(gpu) == 1:
                return _pin_single(
                    gpu_map, gpu[0], hf_params, attention_cls, model_vllm_kwargs, cpu_ram_cfg, sleep_mode
                )
            if replicas == len(gpu) and replicas > 1:
                return _pin_multi_tp1(
                    gpu_map, gpu, hf_params, attention_cls, model_vllm_kwargs, cpu_ram_cfg, sleep_mode
                )
            if replicas != 1:
                raise ValueError(
                    "Pinned gpu=[...] with len>1 is either one TP group (replicas=1) "
                    "or N single-GPU pins (replicas=len(gpu)). One deploy creates at most "
                    "one TP replica — for a second TP copy, call deploy_model again with a "
                    f"different model_id / pin. Got replicas={replicas}, len(gpu)={len(gpu)}."
                )
            return _pin_tp_group(
                gpu_map, gpu, hf_params, attention_cls, model_vllm_kwargs, cpu_ram_cfg, sleep_mode
            )

        assert False, "gpu must be a string, list of strings, or None"

    return _auto_place(
        gpu_map, replicas, hf_params, allocation_cls, attention_cls, model_vllm_kwargs, cpu_ram_cfg, sleep_mode
    )


def _pin_single(gpu_map, gpu_key, hf_params, attention_cls, model_vllm_kwargs, cpu_ram_cfg, sleep_mode=False):
    tp_size = 1
    vram_reqs = build_vram_reqs_for_tp(hf_params, attention_cls, model_vllm_kwargs, tp_size)
    weight_need = fixed_non_kv_gb(vram_reqs, sleep_mode=sleep_mode)
    host = gpu_key.split(":")[0]
    avail = _resolve_pinned_gpu(gpu_map, gpu_key, 0)["available_gb"]
    need = on_gpu_weight_need_gb(
        weight_need, avail, cpu_ram_cfg, host_memory_available_gb(host), 1
    )
    return tp_size, [_resolve_pinned_gpu(gpu_map, gpu_key, need)], vram_reqs


def _pin_multi_tp1(gpu_map, gpu_keys, hf_params, attention_cls, model_vllm_kwargs, cpu_ram_cfg, sleep_mode=False):
    tp_size = 1
    vram_reqs = build_vram_reqs_for_tp(hf_params, attention_cls, model_vllm_kwargs, tp_size)
    weight_need = fixed_non_kv_gb(vram_reqs, sleep_mode=sleep_mode)
    host_counts = count_by_host(gpu_keys)
    resolved = []
    for g in gpu_keys:
        host = g.split(":")[0]
        avail = _resolve_pinned_gpu(gpu_map, g, 0)["available_gb"]
        need = on_gpu_weight_need_gb(
            weight_need, avail, cpu_ram_cfg, host_memory_available_gb(host), host_counts[host]
        )
        resolved.append(_resolve_pinned_gpu(gpu_map, g, need))
    return tp_size, resolved, vram_reqs


def _pin_tp_group(gpu_map, gpu_keys, hf_params, attention_cls, model_vllm_kwargs, cpu_ram_cfg, sleep_mode=False):
    tp_size = len(gpu_keys)
    hosts = {g.split(":")[0] for g in gpu_keys}
    if len(hosts) != 1:
        raise ValueError(f"Same-node TP only — pinned GPUs span hosts {sorted(hosts)}")
    assert_tp_shardable(hf_params, tp_size)
    vram_reqs = build_vram_reqs_for_tp(hf_params, attention_cls, model_vllm_kwargs, tp_size)
    weight_need = fixed_non_kv_gb(vram_reqs, sleep_mode=sleep_mode)
    return tp_size, [_resolve_pinned_gpu(gpu_map, g, weight_need) for g in gpu_keys], vram_reqs


def _auto_place(gpu_map, replicas, hf_params, allocation_cls, attention_cls, model_vllm_kwargs, cpu_ram_cfg, sleep_mode=False):
    if not gpu_map:
        raise ValueError("No GPUs in registry — cannot place model")

    vram_reqs_1 = build_vram_reqs_for_tp(hf_params, attention_cls, model_vllm_kwargs, 1)
    weight_need_1 = fixed_non_kv_gb(vram_reqs_1, sleep_mode=sleep_mode)
    alloc_kwargs_1 = dict(model_vllm_kwargs)
    alloc_kwargs_1["cpu_ram_per_instance"] = cpu_ram_cfg
    single_cls = allocation_cls or RayPerformanceAllocator
    chosen = single_cls().select(
        gpu_map, replicas, weight_need_1, hf_params, alloc_kwargs_1
    )
    if chosen:
        if replicas != -1 and len(chosen) < replicas:
            raise ValueError(
                f"Only {len(chosen)} GPU(s) can hold the model "
                f"(need {weight_need_1:.2f}GB each before CPU spill), "
                f"but replicas={replicas}.\nCluster:\n{gpu_inventory_lines(gpu_map)}"
            )
        return 1, [gpu_info_entry(k, g) for k, g in chosen], vram_reqs_1

    if cpu_ram_cfg != 0:
        raise ValueError(
            f"No single GPU can hold the model (need {weight_need_1:.2f}GB each with CPU spill), "
            f"and cpu_ram_per_instance is not supported with tensor parallelism (TP>1).\n"
            f"Cluster:\n{gpu_inventory_lines(gpu_map)}"
        )

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
        raise ValueError(
            f"Only {n_groups} same-node TP={tp_size} group(s) fit "
            f"(need {min_vram:.2f}GB/GPU on GPU VRAM), but replicas={replicas}.\n"
            f"Cluster:\n{gpu_inventory_lines(gpu_map)}"
        )

    raise ValueError(
        f"No GPU set in the cluster can support this model "
        f"(need >={weight_need_1:.2f}GB on one GPU, or same-node TP=2..{max_tp} on GPU VRAM only).\n"
        f"Cluster:\n{gpu_inventory_lines(gpu_map)}"
    )
