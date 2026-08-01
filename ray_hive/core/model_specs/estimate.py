"""Cluster-agnostic HF config load + VRAM estimate (no Ray)."""
from typing import Optional, Type

from .attention import BaseAttentionSpecs
from .planner import build_vram_reqs, effective_input_len, is_pooling_vram, normalize_hf_config


def load_hf_config_dict(model_name: str) -> dict:
    """
    Load HF model config with nested defaults applied.

    Raw config.json often stores only text_config / audio_config overrides
    (e.g. Qwen2-Audio); AutoConfig merges architecture defaults so planners
    see hidden_size, num_hidden_layers, etc.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return cfg.to_dict()


def estimate_vram(
    model_name: str,
    max_input_prompt_length: int,
    max_output_prompt_length: int = 0,
    max_num_seqs: int = 32,
    max_num_batched_tokens: Optional[int] = None,
    tensor_parallel_size: int = 1,
    attention_cls: Optional[Type[BaseAttentionSpecs]] = None,
    **kwargs,
) -> dict:
    """
    Estimate per-GPU VRAM for a model without a Ray cluster.

    max_model_len = effective_input + output (output ignored for pooling).
    max_num_batched_tokens defaults to max_num_seqs.
    Pass runner="pooling" (or legacy task="embed") in kwargs for embedding models.
    """
    hf_params = normalize_hf_config(load_hf_config_dict(model_name))
    vram_reqs = build_vram_reqs(
        hf_params,
        attention_cls=attention_cls,
        tensor_parallel_size=tensor_parallel_size,
        **kwargs,
    )
    pooling = is_pooling_vram(vram_reqs)
    input_len = effective_input_len(vram_reqs, max_input_prompt_length)
    if pooling:
        max_model_len = input_len
    else:
        max_model_len = input_len + max_output_prompt_length
    if max_num_batched_tokens is None:
        max_num_batched_tokens = max_num_seqs

    weights_gb = vram_reqs.calc_weights_gb()
    overhead_gb = vram_reqs.calc_system_overhead_gb()
    misc_gb = vram_reqs.calc_misc_vram_gb()
    activation_gb = vram_reqs.calc_activation_gb(max_num_batched_tokens)
    kv_cache_gb = vram_reqs.calc_kv_cache_gb(max_model_len, max_num_seqs)
    total_vram_gb = overhead_gb + weights_gb + misc_gb + activation_gb + kv_cache_gb

    return {
        "model_name": model_name,
        "max_num_seqs": max_num_seqs,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "tensor_parallel_size": tensor_parallel_size,
        "pooling": pooling,
        "weights_gb": weights_gb,
        "kv_cache_gb": kv_cache_gb,
        "activation_gb": activation_gb,
        "overhead_gb": overhead_gb,
        "misc_gb": misc_gb,
        "total_vram_gb": total_vram_gb,
    }
