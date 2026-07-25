"""
Deployment planner — pick VramReqs class and solve inverse VRAM problem.

build_vram_reqs builds the BaseVramReqs calculator from HF config.
plan_deployment computes max_num_seqs, batched tokens, and gpu_memory_utilization.
"""
import math
from typing import Optional, Type

from .attention import BaseAttentionSpecs
from .vram_reqs import BaseVramReqs


def normalize_hf_config(hf_config) -> dict:
    """Flatten nested LM configs (text_config, llm_config) for VRAM/TP planning."""
    params = dict(hf_config if isinstance(hf_config, dict) else hf_config.to_dict())
    for nested_key in ("text_config", "llm_config"):
        nested = params.pop(nested_key, None)
        if isinstance(nested, dict):
            params = {**nested, **params}
    return params


def build_vram_reqs(
    hf_config,
    attention_cls: Optional[Type[BaseAttentionSpecs]] = None,
    **kwargs,
) -> BaseVramReqs:
    """Build BaseVramReqs from HF config dict (caller should normalize nested configs)."""
    params = dict(hf_config)
    params.update(kwargs)
    return BaseVramReqs(attention_cls=attention_cls, **params)


def estimate_max_num_batched_tokens(vram_reqs: BaseVramReqs, input_len: int, output_len: int, kv_cache_gb: float) -> int:
    """
    Estimate max_num_batched_tokens from input/output lengths and per-GPU KV budget.

    BT_max = sqrt((P + G) * T_kv) * P / (P + G), rounded to nearest power of 2.
    """
    p = input_len
    g = output_len
    pg = p + g
    assert p > 0 and g > 0, f"input/output lengths must be positive, got {p}/{g}"

    kv_token = vram_reqs.attention.kv_bytes_per_token() / vram_reqs.tp_size
    assert kv_token > 0, "kv_bytes_per_token must be positive"
    t_kv = (kv_cache_gb * (1024 ** 3)) / kv_token
    assert t_kv > 0, f"KV budget must be positive, got kv_cache_gb={kv_cache_gb}"

    bt_max = math.sqrt(pg * t_kv) * (p / pg)
    bt_max = max(1.0, bt_max)
    return max(1, 2 ** round(math.log2(bt_max)))


def plan_deployment(
    vram_reqs: BaseVramReqs,
    vram_budget_gb: float,
    live_total_vram_gb: float,
    max_model_len: int,
    input_len: int,
    output_len: int,
    max_num_batched_tokens_override: int | None = None,
    max_num_seqs_override: int | None = None,
    cpu_kv_offload_gb: float = 0,
    cpu_weight_offload_gb: float = 0,
    live_available_vram_gb: float | None = None,
) -> dict:
    """
    Solve inverse VRAM problem → vLLM deployment settings dict.

    vram_budget_gb is typically available × gpu_budget_frac(tp_size).
    cpu_weight_offload_gb frees GPU weight VRAM (TP=1). cpu_kv_offload_gb is
    unused (kept for plan display); live concurrency is GPU KV only.
    Decode sampling (flashinfer) peaks at logits + masked + softmax outside the
    util pool (~max_num_seqs × vocab × 10 bytes). CUDA graph capture also needs
    ~0.6 GiB outside the pool. util = planned_pool / live_total (vLLM multiplies
    by total card memory); planned pool is capped so util×total fits in live
    available after those reservations, and max_num_seqs is recomputed to match.
    """
    available_vram_gb = vram_budget_gb
    assert available_vram_gb > 0, f"vram_budget_gb must be positive, got {available_vram_gb}"
    assert live_total_vram_gb > 0, f"live_total_vram_gb must be positive, got {live_total_vram_gb}"
    assert cpu_kv_offload_gb >= 0, f"cpu_kv_offload_gb must be >= 0, got {cpu_kv_offload_gb}"
    assert cpu_weight_offload_gb >= 0, f"cpu_weight_offload_gb must be >= 0, got {cpu_weight_offload_gb}"
    live_avail = live_total_vram_gb if live_available_vram_gb is None else live_available_vram_gb

    if max_num_seqs_override is not None and max_num_batched_tokens_override is not None:
        max_num_seqs = max_num_seqs_override
        max_num_batched_tokens = max_num_batched_tokens_override
        non_kv_vram_gb = vram_reqs.calc_non_kv_vram_gb(max_num_batched_tokens) - cpu_weight_offload_gb
        kv_cache_gb = vram_reqs.calc_kv_cache_gb(max_model_len, max_num_seqs)
        total_vram_gb = non_kv_vram_gb + kv_cache_gb
    elif max_num_batched_tokens_override is not None:
        max_num_batched_tokens = max_num_batched_tokens_override
        non_kv_vram_gb = vram_reqs.calc_non_kv_vram_gb(max_num_batched_tokens) - cpu_weight_offload_gb
        kv_cache_gb = available_vram_gb - non_kv_vram_gb
        if kv_cache_gb <= 0:
            raise ValueError(
                f"Model does not fit after non-KV: budget {available_vram_gb:.2f}GB, "
                f"non-KV {non_kv_vram_gb:.2f}GB (tp_size={vram_reqs.tp_size})"
            )
        max_num_seqs = vram_reqs.attention.calc_max_num_seqs_given_kv_cache(
            max_model_len,
            kv_cache_gb * vram_reqs.tp_size,
        )
        total_vram_gb = non_kv_vram_gb + kv_cache_gb
    elif max_num_seqs_override is not None:
        max_num_seqs = max_num_seqs_override
        kv_cache_gb = vram_reqs.calc_kv_cache_gb(max_model_len, max_num_seqs)
        max_num_batched_tokens = estimate_max_num_batched_tokens(
            vram_reqs, input_len, output_len, kv_cache_gb
        )
        non_kv_vram_gb = vram_reqs.calc_non_kv_vram_gb(max_num_batched_tokens) - cpu_weight_offload_gb
        total_vram_gb = non_kv_vram_gb + kv_cache_gb
    else:
        fixed_on_gpu = (
            vram_reqs.calc_system_overhead_gb()
            + vram_reqs.calc_weights_gb()
            + vram_reqs.calc_misc_vram_gb()
            - cpu_weight_offload_gb
        )
        kv_cache_gb_est = available_vram_gb - fixed_on_gpu
        if kv_cache_gb_est <= 0:
            fit_hint = "Need a larger GPU or larger same-node TP pin."
            if vram_reqs.tp_size == 1:
                fit_hint += " Or more cpu_ram_per_instance."
            raise ValueError(
                f"Model does not fit in VRAM budget {available_vram_gb:.2f}GB: "
                f"fixed non-KV on GPU is {fixed_on_gpu:.2f}GB "
                f"(tp_size={vram_reqs.tp_size}, cpu_weight_offload={cpu_weight_offload_gb:.2f}GB). "
                f"{fit_hint}"
            )
        max_num_batched_tokens = estimate_max_num_batched_tokens(
            vram_reqs, input_len, output_len, kv_cache_gb_est
        )
        non_kv_vram_gb = vram_reqs.calc_non_kv_vram_gb(max_num_batched_tokens) - cpu_weight_offload_gb
        kv_cache_gb = available_vram_gb - non_kv_vram_gb
        if kv_cache_gb <= 0:
            raise ValueError(
                f"Model does not fit after activations: budget {available_vram_gb:.2f}GB, "
                f"non-KV {non_kv_vram_gb:.2f}GB (tp_size={vram_reqs.tp_size})"
            )
        max_num_seqs = vram_reqs.attention.calc_max_num_seqs_given_kv_cache(
            max_model_len,
            kv_cache_gb * vram_reqs.tp_size,
        )
        total_vram_gb = non_kv_vram_gb + kv_cache_gb

    if max_num_batched_tokens < max_num_seqs:
        max_num_batched_tokens = max_num_seqs
        non_kv_vram_gb = vram_reqs.calc_non_kv_vram_gb(max_num_batched_tokens) - cpu_weight_offload_gb
        kv_cache_gb = available_vram_gb - non_kv_vram_gb
        if kv_cache_gb <= 0:
            raise ValueError(
                f"Model does not fit after raising max_num_batched_tokens to {max_num_seqs}: "
                f"budget {available_vram_gb:.2f}GB, non-KV {non_kv_vram_gb:.2f}GB"
            )
        max_num_seqs = vram_reqs.attention.calc_max_num_seqs_given_kv_cache(
            max_model_len,
            kv_cache_gb * vram_reqs.tp_size,
        )
        total_vram_gb = non_kv_vram_gb + kv_cache_gb

    # Outside the util pool during decode sample (flashinfer path):
    #   logits (~2B) + masked_logits (fp32, 4B) + softmax probs (fp32, 4B)
    #   ≈ max_num_seqs × vocab × 10 bytes (matches warmup OOM size).
    # CUDA graph capture also sits outside the util pool (~0.58 GiB observed).
    vocab = float(vram_reqs.hf_params["vocab_size"])
    bytes_per_seq_vocab = 10.0
    graph_headroom_gb = 0.6
    device_gb = live_total_vram_gb
    usable_gb = min(device_gb, live_avail)

    for _ in range(2):
        sample_scratch_gb = max_num_seqs * vocab * bytes_per_seq_vocab / (1024 ** 3)
        max_pool_gb = usable_gb - sample_scratch_gb - graph_headroom_gb
        if max_pool_gb <= non_kv_vram_gb:
            room_gb = usable_gb - non_kv_vram_gb - graph_headroom_gb
            per_seq = vocab * bytes_per_seq_vocab / (1024 ** 3)
            max_num_seqs = max(1, int(room_gb / per_seq)) if room_gb > 0 and per_seq > 0 else 1
            if max_num_batched_tokens < max_num_seqs:
                max_num_batched_tokens = max_num_seqs
            non_kv_vram_gb = vram_reqs.calc_non_kv_vram_gb(max_num_batched_tokens) - cpu_weight_offload_gb
            continue
        if total_vram_gb > max_pool_gb:
            total_vram_gb = max_pool_gb
            kv_cache_gb = total_vram_gb - non_kv_vram_gb
            if kv_cache_gb <= 0:
                raise ValueError(
                    f"No KV room after sampler/graph headroom: pool {max_pool_gb:.2f}GB, "
                    f"non-KV {non_kv_vram_gb:.2f}GB"
                )
            max_num_seqs = vram_reqs.attention.calc_max_num_seqs_given_kv_cache(
                max_model_len,
                kv_cache_gb * vram_reqs.tp_size,
            )
            if max_num_batched_tokens < max_num_seqs:
                max_num_batched_tokens = max_num_seqs
            continue
        break

    gpu_memory_utilization = total_vram_gb / device_gb

    return {
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
        "total_vram_gb": total_vram_gb,
        "cpu_kv_offload_gb": cpu_kv_offload_gb,
        "cpu_offload_gb": cpu_weight_offload_gb,
    }
