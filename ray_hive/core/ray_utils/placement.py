"""GPU placement helpers used by deploy planning."""
from ray_hive.core.gpu_alloc import TP1_BUDGET_FRAC
from ray_hive.core.model_specs.factory import is_multimodal_hf, resolve_limit_mm_per_prompt
from ray_hive.core.model_specs.planner import (
    build_vram_reqs,
    effective_input_len,
    is_pooling_vram,
    plan_deployment,
)
from ray_hive.errors import (
    ConfigError,
    InsufficientVramError,
    KvBudgetError,
    NoPlacementError,
    PlacementError,
)

from .naming import deployment_name

# Text-token floor when max_input_prompt_length="auto" (output stays user-fixed).
AUTO_INPUT_FLOOR = 256
# Safety cap when HF config has no max_position_embeddings / model_max_length.
AUTO_INPUT_ABS_CAP = 1 << 20


def is_auto_input(value) -> bool:
    """True when max_input_prompt_length is the auto sentinel."""
    return isinstance(value, str) and value == "auto"


def validate_auto_input_config(config: dict) -> None:
    """
    Require max_num_seqs when max_input_prompt_length=\"auto\".

    Auto grows text input only (floor AUTO_INPUT_FLOOR); output length stays fixed.
    """
    text_in = config.get("max_input_prompt_length")
    if not is_auto_input(text_in):
        return
    if config.get("max_num_seqs") is None:
        raise ConfigError(
            'max_input_prompt_length="auto" requires max_num_seqs '
            "(pass it inside vllm_kwargs)"
        )
    if int(config["max_num_seqs"]) < 1:
        raise ConfigError("max_num_seqs must be >= 1 when using max_input_prompt_length=\"auto\"")


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
    text_in = config["max_input_prompt_length"]
    if is_auto_input(text_in):
        raise ConfigError(
            "plan_lengths needs a concrete max_input_prompt_length; resolve \"auto\" first"
        )
    if not isinstance(text_in, int) or text_in <= 0:
        raise ConfigError(
            f"max_input_prompt_length must be a positive int or \"auto\", got {text_in!r}"
        )
    pooling = is_pooling_vram(vram_reqs)
    text_out = config["max_output_prompt_length"]
    input_len = effective_input_len(vram_reqs, text_in)
    if pooling:
        return input_len, 0, input_len, True
    return input_len, text_out, input_len + text_out, False


def lengths_for_text_input(
    vram_reqs, text_in: int, text_out: int
) -> tuple[int, int, int, bool]:
    """Return (effective_input, output_len, max_model_len, pooling) for a text input length."""
    pooling = is_pooling_vram(vram_reqs)
    input_len = effective_input_len(vram_reqs, text_in)
    if pooling:
        return input_len, 0, input_len, True
    return input_len, text_out, input_len + text_out, False


def _hf_text_input_cap(hf_params: dict, vram_reqs, text_out: int, pooling: bool) -> int | None:
    """Largest text input that keeps max_model_len within HF position limit, if known."""
    raw = hf_params.get("max_position_embeddings")
    if raw is None:
        raw = hf_params.get("model_max_length")
    if raw is None:
        return None
    mm = int(vram_reqs.attention.mm_tokens_per_prompt())
    hi = int(raw)
    if pooling:
        return max(0, hi - mm)
    return max(0, hi - mm - max(0, text_out))


def _try_plan_at_text_input(
    vram_reqs,
    text_in: int,
    text_out: int,
    *,
    vram_budget_gb: float,
    live_total_vram_gb: float,
    live_available_vram_gb: float,
    max_num_seqs: int,
    max_num_batched_tokens_override: int | None,
    sleep_mode: bool,
    enforce_eager: bool,
) -> tuple[dict, int, int, int]:
    """Plan at a concrete text input. Returns (plan, text_in, input_len, max_model_len)."""
    input_len, output_len, max_model_len, pooling = lengths_for_text_input(
        vram_reqs, text_in, text_out
    )
    plan = plan_deployment(
        vram_reqs,
        vram_budget_gb=vram_budget_gb,
        live_total_vram_gb=live_total_vram_gb,
        max_model_len=max_model_len,
        input_len=input_len,
        output_len=max(1, output_len) if not pooling else 0,
        max_num_batched_tokens_override=max_num_batched_tokens_override,
        max_num_seqs_override=max_num_seqs,
        live_available_vram_gb=live_available_vram_gb,
        sleep_mode=sleep_mode,
        pooling=pooling,
        enforce_eager=enforce_eager,
    )
    # Auto mode fixes concurrency; reject candidates where the planner had to
    # shrink max_num_seqs to make a too-large context fit.
    if int(plan["max_num_seqs"]) < int(max_num_seqs):
        raise KvBudgetError(
            f"Cannot honor max_num_seqs={max_num_seqs} at text_in={text_in} "
            f"(planner packed {plan['max_num_seqs']})"
        )
    return plan, text_in, input_len, max_model_len


def solve_auto_text_input(
    vram_reqs,
    hf_params: dict,
    text_out: int,
    *,
    vram_budget_gb: float,
    live_total_vram_gb: float,
    live_available_vram_gb: float,
    max_num_seqs: int,
    max_num_batched_tokens_override: int | None = None,
    sleep_mode: bool = False,
    enforce_eager: bool = False,
) -> tuple[dict, int, int, int]:
    """
    Largest text input (>= AUTO_INPUT_FLOOR) that fits at fixed max_num_seqs.

    Output length is fixed. Effective input includes MM placeholders, so auto
    never drops below what MmContextError would require for those placeholders.
    """
    floor = AUTO_INPUT_FLOOR
    hf_cap = _hf_text_input_cap(hf_params, vram_reqs, text_out, is_pooling_vram(vram_reqs))
    abs_hi = AUTO_INPUT_ABS_CAP if hf_cap is None else min(AUTO_INPUT_ABS_CAP, hf_cap)
    if abs_hi < floor:
        raise ConfigError(
            f"HF context cap ({abs_hi}) is below auto input floor ({floor}) "
            f"after MM placeholders / output reservation"
        )

    def _at(n: int):
        return _try_plan_at_text_input(
            vram_reqs,
            n,
            text_out,
            vram_budget_gb=vram_budget_gb,
            live_total_vram_gb=live_total_vram_gb,
            live_available_vram_gb=live_available_vram_gb,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens_override=max_num_batched_tokens_override,
            sleep_mode=sleep_mode,
            enforce_eager=enforce_eager,
        )

    # Floor must fit; MM placeholders are already folded into effective input.
    best = _at(floor)

    lo = floor
    hi = floor
    while True:
        nxt = min(hi * 2 if hi > 0 else floor * 2, abs_hi)
        if nxt <= hi:
            break
        try:
            best = _at(nxt)
            lo = nxt
            hi = nxt
        except ValueError:
            # Binary-search the failing upper bound.
            fail_hi = nxt
            while lo + 1 < fail_hi:
                mid = (lo + fail_hi) // 2
                try:
                    best = _at(mid)
                    lo = mid
                except ValueError:
                    fail_hi = mid
            return best

    # Hit HF / abs cap while still fitting.
    return best


def _annotate_plan(
    plan: dict,
    *,
    tp_size: int,
    weight_need: float,
    weights_gb: float,
    text_in: int,
    text_out: int,
    max_model_len: int,
    auto_input: bool,
) -> dict:
    plan = dict(plan)
    plan["tensor_parallel_size"] = tp_size
    plan["weights_gb"] = weights_gb
    plan["weight_need_gb"] = weight_need
    plan["max_input_prompt_length"] = text_in
    plan["max_output_prompt_length"] = text_out
    plan["max_model_len"] = max_model_len
    plan["max_input_prompt_length_auto"] = auto_input
    return plan


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

    validate_auto_input_config(config)
    auto_input = is_auto_input(config.get("max_input_prompt_length"))

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

    text_out = int(config["max_output_prompt_length"])
    if not auto_input:
        input_len, output_len, max_model_len, pooling = plan_lengths(vram_reqs, config)
        text_in = int(config["max_input_prompt_length"])
    else:
        # Resolved per GPU group below; placeholders for weight gate only.
        pooling = is_pooling_vram(vram_reqs)
        text_in = AUTO_INPUT_FLOOR
        input_len, output_len, max_model_len, pooling = lengths_for_text_input(
            vram_reqs, text_in, text_out
        )

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
            if auto_input:
                plan, text_in, input_len, max_model_len = solve_auto_text_input(
                    vram_reqs,
                    hf_params,
                    text_out,
                    vram_budget_gb=per_gpu_budget,
                    live_total_vram_gb=bottleneck["total_gb"],
                    live_available_vram_gb=avail,
                    max_num_seqs=int(config["max_num_seqs"]),
                    max_num_batched_tokens_override=config.get("max_num_batched_tokens"),
                    sleep_mode=sleep_mode,
                    enforce_eager=enforce_eager,
                )
                output_len = 0 if pooling else text_out
            else:
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
        except ConfigError:
            raise
        except ValueError:
            # replicas=-1: pack every GPU that fits; skip cards too small for the plan.
            if replicas != -1:
                raise
            continue

        plan = _annotate_plan(
            plan,
            tp_size=tp_size,
            weight_need=weight_need,
            weights_gb=vram_reqs.calc_weights_gb() * tp_size,
            text_in=text_in,
            text_out=0 if pooling else text_out,
            max_model_len=max_model_len,
            auto_input=auto_input,
        )

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
