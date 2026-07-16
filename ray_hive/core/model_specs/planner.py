"""
Deployment planner — pick VramReqs class and solve inverse VRAM problem.

build_vram_reqs selects the calculator from HF config.
plan_deployment computes max_num_seqs, batched tokens, and gpu_memory_utilization.
"""
import math

from .vram_reqs import BaseVramReqs, Qwen35VramReqs


def build_vram_reqs(hf_config, **kwargs) -> BaseVramReqs:
    """Pick VramReqs class from HF config and build calculator."""
    params = hf_config if isinstance(hf_config, dict) else hf_config.to_dict()
    params.update(kwargs)

    if params.get("num_attention_layers") is not None:
        return Qwen35VramReqs(**params)
    return BaseVramReqs(**params)


def estimate_max_num_batched_tokens(vram_reqs: BaseVramReqs, input_len: int, output_len: int, kv_cache_gb: float) -> int:
    """
    Estimate max_num_batched_tokens from input/output lengths and KV budget.

    BT_max = sqrt((P + G) * T_kv) * P / (P + G), rounded to nearest power of 2.
    """
    p = input_len
    g = output_len
    pg = p + g

    kv_token = vram_reqs.attention.kv_bytes_per_token()
    t_kv = (kv_cache_gb * (1024 ** 3)) / kv_token

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
) -> dict:
    """
    Solve inverse VRAM problem → vLLM deployment settings dict.

    max_num_seqs and max_num_batched_tokens are estimated by default.
    Either can be overridden independently; both can be set together.

    vram_budget_gb is the caller's planning budget (typically
    (available_gb - deployment_used_gb) * 0.95) — max_num_seqs and KV
    cache sizing are computed against that budget, not raw free VRAM.
    """
    available_vram_gb = vram_budget_gb

    if max_num_seqs_override is not None and max_num_batched_tokens_override is not None:
        max_num_seqs = max_num_seqs_override
        max_num_batched_tokens = max_num_batched_tokens_override
        non_kv_vram_gb = vram_reqs.calc_non_kv_vram_gb(max_num_batched_tokens)
        kv_cache_gb = vram_reqs.calc_kv_cache_gb(max_model_len, max_num_seqs)
        total_vram_gb = non_kv_vram_gb + kv_cache_gb
    elif max_num_batched_tokens_override is not None:
        max_num_batched_tokens = max_num_batched_tokens_override
        non_kv_vram_gb = vram_reqs.calc_non_kv_vram_gb(max_num_batched_tokens)
        kv_cache_gb = available_vram_gb - non_kv_vram_gb
        max_num_seqs = vram_reqs.attention.calc_max_num_seqs_given_kv_cache(max_model_len, kv_cache_gb)
        total_vram_gb = non_kv_vram_gb + kv_cache_gb
    elif max_num_seqs_override is not None:
        max_num_seqs = max_num_seqs_override
        kv_cache_gb = vram_reqs.calc_kv_cache_gb(max_model_len, max_num_seqs)
        max_num_batched_tokens = estimate_max_num_batched_tokens(
            vram_reqs,
            input_len,
            output_len,
            kv_cache_gb,
        )
        non_kv_vram_gb = vram_reqs.calc_non_kv_vram_gb(max_num_batched_tokens)
        total_vram_gb = non_kv_vram_gb + kv_cache_gb
    else:
        fixed_non_kv_gb = (
            vram_reqs.calc_system_overhead_gb()
            + vram_reqs.calc_weights_gb()
            + vram_reqs.calc_misc_vram_gb()
        )
        kv_cache_gb_est = available_vram_gb - fixed_non_kv_gb
        max_num_batched_tokens = estimate_max_num_batched_tokens(
            vram_reqs,
            input_len,
            output_len,
            kv_cache_gb_est,
        )
        non_kv_vram_gb = vram_reqs.calc_non_kv_vram_gb(max_num_batched_tokens)
        kv_cache_gb = available_vram_gb - non_kv_vram_gb
        max_num_seqs = vram_reqs.attention.calc_max_num_seqs_given_kv_cache(max_model_len, kv_cache_gb)
        total_vram_gb = non_kv_vram_gb + kv_cache_gb

    gpu_memory_utilization = total_vram_gb / live_total_vram_gb

    return {
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
        "kv_cache_gb": kv_cache_gb,
        "non_kv_vram_gb": non_kv_vram_gb,
        "total_vram_gb": total_vram_gb,
    }
