"""GPU placement helpers used by deploy planning."""
from ray_hive.core.gpu_alloc import TP1_BUDGET_FRAC
from ray_hive.core.model_specs.factory import is_multimodal_hf, resolve_limit_mm_per_prompt
from ray_hive.core.model_specs.planner import (
    build_vram_reqs,
    effective_input_len,
    is_pooling_vram,
    plan_deployment,
)
from ray_hive.errors import InsufficientVramError, NoPlacementError, PlacementError

from .naming import deployment_name


def fixed_non_kv_gb(vram_reqs, sleep_mode: bool = False) -> float:
    """Minimum per-GPU VRAM to load weights + overhead (before KV)."""
    return vram_reqs.calc_fixed_non_kv_gb(sleep_mode)


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
        raise PlacementError(
            f"Got {len(gpus)} GPUs but tensor_parallel_size={tp_size} "
            f"(need a multiple of {tp_size})"
        )
    return [gpus[i : i + tp_size] for i in range(0, len(gpus), tp_size)]


def plan_lengths(vram_reqs, config: dict) -> tuple[int, int, int, bool]:
    """Return (input_len, output_len, max_model_len, pooling) from deploy config."""
    pooling = is_pooling_vram(vram_reqs)
    text_in = config["max_input_prompt_length"]
    text_out = config["max_output_prompt_length"]
    input_len = effective_input_len(vram_reqs, text_in)
    if pooling:
        return input_len, 0, input_len, True
    return input_len, text_out, input_len + text_out, False


def plan_replica_groups(
    gpu_map: dict,
    config: dict,
    hf_params: dict,
    model_vllm_kwargs: dict,
    model_id: str = "estimate",
) -> dict:
    """
    Dry-run the same packing deploy uses. Returns
    {replica_id: {plan, gpu_keys, group, tp_size, max_model_len}}.
    """
    from .select_gpus import resolve_target_gpus

    sleep_mode = float(config.get("sleep_timeout", -1) or -1) > 0
    enforce_eager = bool(model_vllm_kwargs.get("enforce_eager", False))
    if is_multimodal_hf(hf_params):
        model_vllm_kwargs.setdefault(
            "limit_mm_per_prompt",
            resolve_limit_mm_per_prompt(hf_params, model_vllm_kwargs),
        )

    tp_size, target_gpus, vram_reqs = resolve_target_gpus(
        gpu_map,
        config.get("replicas", -1),
        config.get("gpu"),
        hf_params,
        config.get("allocation_cls"),
        config.get("attention_cls"),
        model_vllm_kwargs,
        sleep_mode=sleep_mode,
    )
    gpu_groups = chunk_gpu_groups(target_gpus, tp_size)

    input_len, output_len, max_model_len, pooling = plan_lengths(vram_reqs, config)
    weight_need = fixed_non_kv_gb(vram_reqs, sleep_mode=sleep_mode)
    replicas = config.get("replicas", -1)
    results = {}

    for group in gpu_groups:
        gpu_keys = [g["gpu_key"] for g in group]
        bottleneck = min(group, key=lambda g: g["available_gb"])
        avail = min(g["available_gb"] for g in group)
        device = min(g["total_gb"] for g in group)
        if weight_need > device * TP1_BUDGET_FRAC:
            if replicas != -1:
                raise InsufficientVramError(
                    f"GPU(s) {gpu_keys} util capacity {device * TP1_BUDGET_FRAC:.2f}GB "
                    f"(total {device:.2f}GB × {TP1_BUDGET_FRAC}) < weight need "
                    f"{weight_need:.2f}GB",
                    need_gb=weight_need,
                )
            continue
        per_gpu_budget = avail * TP1_BUDGET_FRAC
        try:
            plan = plan_deployment(
                vram_reqs,
                vram_budget_gb=per_gpu_budget,
                live_total_vram_gb=bottleneck["total_gb"],
                max_model_len=max_model_len,
                input_len=input_len,
                output_len=max(1, output_len) if not pooling else 0,
                max_num_batched_tokens_override=config.get("max_num_batched_tokens"),
                max_num_seqs_override=config.get("max_num_seqs"),
                live_available_vram_gb=avail,
                sleep_mode=sleep_mode,
                pooling=pooling,
                enforce_eager=enforce_eager,
            )
        except ValueError:
            # replicas=-1: pack every GPU that fits; skip cards too small for the plan.
            if replicas != -1:
                raise
            continue
        plan["tensor_parallel_size"] = tp_size
        plan["weights_gb"] = vram_reqs.calc_weights_gb() * tp_size
        plan["weight_need_gb"] = weight_need

        replica_id = deployment_name(model_id, gpu_keys)
        results[replica_id] = {
            "plan": plan,
            "gpu_keys": gpu_keys,
            "group": group,
            "tp_size": tp_size,
            "max_model_len": max_model_len,
        }

    if not results:
        raise NoPlacementError(
            f"No GPU group can fit this model after packing "
            f"(need >={weight_need:.2f}GB fixed non-KV per GPU in the util budget)."
        )
    return results
