"""
Deployment planner — pick VramReqs class and solve inverse VRAM problem.

build_vram_reqs selects the VramReqs/attention family from HF + vllm kwargs.
plan_deployment computes max_num_seqs, batched tokens, and gpu_memory_utilization.
"""
import math
from typing import Optional, Type

from ray_hive.errors import KvBudgetError, MmContextError, ModelDoesNotFitError

from .attention import BaseAttentionSpecs
from .factory import (
    is_multimodal_hf,
    is_pooling_kwargs,
    resolve_limit_mm_per_prompt,
    select_vram_classes,
)
from .vram_reqs import BaseVramReqs


def normalize_hf_config(hf_config) -> dict:
    """
    Flatten nested LM configs (text_config, llm_config) for VRAM/TP planning.

    Retains vision_config / audio_config and other top-level multimodal fields.
    """
    params = dict(hf_config if isinstance(hf_config, dict) else hf_config.to_dict())
    # Keep nested modality configs before flattening LM fields over the top.
    vision_config = params.get("vision_config")
    audio_config = params.get("audio_config")
    for nested_key in ("text_config", "llm_config"):
        nested = params.pop(nested_key, None)
        if isinstance(nested, dict):
            params = {**nested, **params}
    if isinstance(vision_config, dict):
        params["vision_config"] = vision_config
    if isinstance(audio_config, dict):
        params["audio_config"] = audio_config
    return params


def build_vram_reqs(
    hf_config,
    attention_cls: Optional[Type[BaseAttentionSpecs]] = None,
    **kwargs,
) -> BaseVramReqs:
    """Build the appropriate VramReqs subclass from HF config + deploy kwargs."""
    params = dict(hf_config)
    vllm_like = dict(kwargs)
    attn_cls, vram_cls = select_vram_classes(params, attention_cls=attention_cls)

    if is_multimodal_hf(params):
        vllm_like.setdefault(
            "limit_mm_per_prompt",
            resolve_limit_mm_per_prompt(params, vllm_like),
        )

    params.update(vllm_like)
    vram_reqs = vram_cls(attention_cls=attn_cls, **params)
    vram_reqs.pooling = is_pooling_kwargs(vllm_like)
    return vram_reqs


def effective_input_len(vram_reqs: BaseVramReqs, text_input_len: int) -> int:
    """Text input length plus MM placeholder tokens when applicable."""
    return int(vram_reqs.attention.effective_input_len(text_input_len))


def is_pooling_vram(vram_reqs: BaseVramReqs) -> bool:
    return bool(vram_reqs.pooling)


def estimate_max_num_batched_tokens(
    vram_reqs: BaseVramReqs,
    input_len: int,
    output_len: int,
    kv_cache_gb: float,
) -> int:
    """
    Estimate max_num_batched_tokens from input/output lengths and per-GPU KV budget.

    Generate: BT ≈ ((P+G)·T_kv)^(1/3)·P/(P+G). Pooling: BT ≈ (P·T_kv)^(1/3).
    Cube-root in T_kv softens growth when KV budget is large.
    Rounded to nearest power of 2, then raised to any MM item floor.
    """
    p = input_len
    assert p > 0, f"input_len must be positive, got {p}"
    kv_token = vram_reqs.attention.kv_bytes_per_token()
    assert kv_token > 0, "kv_bytes_per_token must be positive"
    t_kv = (kv_cache_gb * (1024 ** 3)) / kv_token
    assert t_kv > 0, f"KV budget must be positive, got kv_cache_gb={kv_cache_gb}"

    if output_len > 0:
        pg = p + output_len
        bt_max = (pg * t_kv) ** (1.0 / 3.0) * (p / pg)
    else:
        bt_max = (p * t_kv) ** (1.0 / 3.0)

    bt = max(1, 2 ** round(math.log2(max(1.0, bt_max))))

    mm_tok = int(vram_reqs.attention.mm_tokens_per_prompt())
    mm_item = int(vram_reqs.attention.max_tokens_per_mm_item())
    floor = max(mm_tok, mm_item)
    if floor > 0:
        # Prefix-LM MM (e.g. Gemma4) needs BT >= max_tokens_per_mm_item.
        bt = max(bt, floor)
    return max(1, bt)


def plan_deployment(
    vram_reqs: BaseVramReqs,
    vram_budget_gb: float,
    live_total_vram_gb: float,
    max_model_len: int,
    input_len: int,
    output_len: int,
    max_num_batched_tokens_override: int | None = None,
    max_num_seqs_override: int | None = None,
    live_available_vram_gb: float | None = None,
    sleep_mode: bool = False,
    pooling: bool | None = None,
    enforce_eager: bool = False,
) -> dict:
    """
    Solve inverse VRAM problem → vLLM deployment settings dict.

    One forward pass: fixed non-KV → BT → seqs (joint with outside-pool sampler/
    graphs) → util, with a single freemem util rescale when needed.
    """
    if pooling is None:
        pooling = is_pooling_vram(vram_reqs)
    else:
        vram_reqs.pooling = pooling

    available_vram_gb = vram_budget_gb
    assert available_vram_gb > 0, f"vram_budget_gb must be positive, got {available_vram_gb}"
    assert live_total_vram_gb > 0, f"live_total_vram_gb must be positive, got {live_total_vram_gb}"
    live_avail = live_total_vram_gb if live_available_vram_gb is None else live_available_vram_gb
    device_gb = live_total_vram_gb
    usable_gb = min(device_gb, live_avail)

    mm_tok = int(vram_reqs.attention.mm_tokens_per_prompt())
    if mm_tok > 0 and max_model_len < mm_tok + (0 if pooling else max(1, output_len)):
        raise MmContextError(
            f"max_model_len={max_model_len} cannot cover MM placeholders ({mm_tok}) "
            f"+ output ({0 if pooling else output_len}). Raise max_input_prompt_length "
            f"or lower limit_mm_per_prompt."
        )

    out_for_bt = 0 if pooling else output_len

    def _estimate_bt(kv_gb: float) -> int:
        return estimate_max_num_batched_tokens(
            vram_reqs, input_len, out_for_bt, kv_gb
        )

    def _non_kv(bt: int) -> float:
        return vram_reqs.calc_non_kv_vram_gb(bt, sleep_mode=sleep_mode)

    fixed_on_gpu = vram_reqs.calc_fixed_non_kv_gb(sleep_mode)
    min_kv_gb = vram_reqs.calc_kv_cache_gb(max_model_len, 1)
    graph_gb = vram_reqs.calc_cuda_graph_gb(
        enforce_eager, usable_gb=usable_gb, min_pool_gb=fixed_on_gpu + min_kv_gb
    )

    if max_num_seqs_override is not None and max_num_batched_tokens_override is not None:
        max_num_seqs = max_num_seqs_override
        max_num_batched_tokens = max_num_batched_tokens_override
        non_kv_vram_gb = _non_kv(max_num_batched_tokens)
        kv_cache_gb = vram_reqs.calc_kv_cache_gb(max_model_len, max_num_seqs)
        total_vram_gb = non_kv_vram_gb + kv_cache_gb
    elif max_num_batched_tokens_override is not None:
        max_num_batched_tokens = max_num_batched_tokens_override
        non_kv_vram_gb = _non_kv(max_num_batched_tokens)
        kv_cache_gb = available_vram_gb - non_kv_vram_gb
        if kv_cache_gb <= 0:
            raise ModelDoesNotFitError(
                f"Model does not fit after non-KV: budget {available_vram_gb:.2f}GB, "
                f"non-KV {non_kv_vram_gb:.2f}GB (tp_size={vram_reqs.tp_size})"
            )
        max_num_seqs = vram_reqs.attention.calc_max_num_seqs_given_kv_cache(
            max_model_len, kv_cache_gb
        )
        max_num_seqs = min(max_num_seqs, max_num_batched_tokens)
        total_vram_gb = non_kv_vram_gb + kv_cache_gb
    elif max_num_seqs_override is not None:
        max_num_seqs = max_num_seqs_override
        kv_cache_gb = vram_reqs.calc_kv_cache_gb(max_model_len, max_num_seqs)
        max_num_batched_tokens = _estimate_bt(kv_cache_gb)
        if mm_tok > 0:
            max_num_seqs = min(max_num_seqs, max_num_batched_tokens)
            kv_cache_gb = vram_reqs.calc_kv_cache_gb(max_model_len, max_num_seqs)
        elif max_num_batched_tokens < max_num_seqs:
            max_num_batched_tokens = max_num_seqs
        non_kv_vram_gb = _non_kv(max_num_batched_tokens)
        total_vram_gb = non_kv_vram_gb + kv_cache_gb
    else:
        if fixed_on_gpu >= available_vram_gb:
            raise ModelDoesNotFitError(
                f"Model does not fit in VRAM budget {available_vram_gb:.2f}GB: "
                f"fixed non-KV on GPU is {fixed_on_gpu:.2f}GB "
                f"(tp_size={vram_reqs.tp_size}). "
                f"Need a larger GPU or larger same-node TP pin."
            )
        max_num_batched_tokens = _estimate_bt(available_vram_gb - fixed_on_gpu)
        non_kv_vram_gb = _non_kv(max_num_batched_tokens)
        if non_kv_vram_gb >= available_vram_gb:
            raise ModelDoesNotFitError(
                f"Model does not fit after activations: budget {available_vram_gb:.2f}GB, "
                f"non-KV {non_kv_vram_gb:.2f}GB (tp_size={vram_reqs.tp_size})"
            )

        def _seqs_for_room(non_kv: float) -> int:
            room_bytes = (usable_gb - graph_gb - non_kv) * (1024 ** 3)
            kv_per_seq = vram_reqs.attention.kv_bytes_per_sequence(max_model_len)
            per_seq = kv_per_seq + vram_reqs.sampler_bytes_per_seq()
            n = max(1, int(room_bytes / per_seq)) if room_bytes > 0 and per_seq > 0 else 1
            kv_from_budget = available_vram_gb - non_kv
            return min(
                n,
                vram_reqs.attention.calc_max_num_seqs_given_kv_cache(max_model_len, kv_from_budget),
            )

        max_num_seqs = _seqs_for_room(non_kv_vram_gb)
        if mm_tok > 0:
            # BT already MM-capped; concurrency cannot exceed the batch window.
            max_num_seqs = min(max_num_seqs, max_num_batched_tokens)
        elif max_num_batched_tokens < max_num_seqs:
            max_num_batched_tokens = max_num_seqs
            non_kv_vram_gb = _non_kv(max_num_batched_tokens)
            if non_kv_vram_gb >= available_vram_gb:
                raise ModelDoesNotFitError(
                    f"Model does not fit after raising max_num_batched_tokens to {max_num_seqs}: "
                    f"budget {available_vram_gb:.2f}GB, non-KV {non_kv_vram_gb:.2f}GB"
                )
            max_num_seqs = min(max_num_seqs, _seqs_for_room(non_kv_vram_gb), max_num_batched_tokens)
        kv_cache_gb = vram_reqs.calc_kv_cache_gb(max_model_len, max_num_seqs)
        total_vram_gb = non_kv_vram_gb + kv_cache_gb

    # Cap util pool so outside-pool graphs+sampler still fit in usable.
    outside_gb = graph_gb + vram_reqs.calc_sampler_scratch_gb(max_num_seqs)
    max_pool_gb = usable_gb - outside_gb
    if total_vram_gb > max_pool_gb:
        if max_pool_gb <= non_kv_vram_gb:
            raise KvBudgetError(
                f"No KV room after graph/sampler headroom: pool {max_pool_gb:.2f}GB, "
                f"non-KV {non_kv_vram_gb:.2f}GB"
            )
        total_vram_gb = max_pool_gb
        kv_cache_gb = total_vram_gb - non_kv_vram_gb
        max_num_seqs = vram_reqs.attention.calc_max_num_seqs_given_kv_cache(
            max_model_len, kv_cache_gb
        )
        if mm_tok > 0:
            max_num_seqs = min(max_num_seqs, max_num_batched_tokens)
        elif max_num_batched_tokens < max_num_seqs:
            max_num_batched_tokens = max_num_seqs
            non_kv_vram_gb = _non_kv(max_num_batched_tokens)
            kv_cache_gb = total_vram_gb - non_kv_vram_gb
            if kv_cache_gb < min_kv_gb:
                raise KvBudgetError(
                    f"No KV room after BT raise for seqs: pool {total_vram_gb:.2f}GB, "
                    f"non-KV {non_kv_vram_gb:.2f}GB"
                )
            max_num_seqs = min(
                max_num_seqs,
                vram_reqs.attention.calc_max_num_seqs_given_kv_cache(max_model_len, kv_cache_gb),
                max_num_batched_tokens,
            )
        kv_cache_gb = vram_reqs.calc_kv_cache_gb(max_model_len, max_num_seqs)
        total_vram_gb = non_kv_vram_gb + kv_cache_gb

    gpu_memory_utilization = total_vram_gb / device_gb
    # vLLM requires free >= util * device_total at startup.
    if live_avail < device_gb and device_gb > 0:
        max_util = live_avail / device_gb
        if gpu_memory_utilization > max_util:
            pool_gb = max_util * device_gb
            if pool_gb < fixed_on_gpu + min_kv_gb:
                raise ModelDoesNotFitError(
                    f"Free VRAM {live_avail:.2f}GB cannot cover fixed non-KV "
                    f"{fixed_on_gpu:.2f}GB + min KV {min_kv_gb:.2f}GB"
                )
            gpu_memory_utilization = max_util
            total_vram_gb = pool_gb
            non_kv_vram_gb = _non_kv(max_num_batched_tokens)
            kv_cache_gb = total_vram_gb - non_kv_vram_gb
            if kv_cache_gb < min_kv_gb:
                raise KvBudgetError(
                    f"No KV room after freemem util clamp: pool {total_vram_gb:.2f}GB, "
                    f"non-KV {non_kv_vram_gb:.2f}GB"
                )
            max_num_seqs = vram_reqs.attention.calc_max_num_seqs_given_kv_cache(
                max_model_len, kv_cache_gb
            )
            max_num_seqs = min(max_num_seqs, max_num_batched_tokens)

    return {
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
        "total_vram_gb": total_vram_gb,
        "pooling": pooling,
        "mm_tokens_per_prompt": mm_tok,
    }
