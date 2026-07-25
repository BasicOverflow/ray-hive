"""Cluster-agnostic HF config load + VRAM estimate (no Ray)."""
import json
from pathlib import Path
from typing import Optional, Type

from .attention import BaseAttentionSpecs
from .planner import build_vram_reqs, normalize_hf_config


def load_hf_config_dict(model_name: str) -> dict:
    """Load HF config.json from a local path or the Hub (no AutoConfig)."""
    local = Path(model_name) / "config.json"
    if local.is_file():
        return json.loads(local.read_text())

    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=model_name, filename="config.json")
    return json.loads(Path(path).read_text())


def estimate_vram(
    model_name: str,
    max_input_prompt_length: int,
    max_output_prompt_length: int,
    max_num_seqs: int = 32,
    max_num_batched_tokens: Optional[int] = None,
    tensor_parallel_size: int = 1,
    attention_cls: Optional[Type[BaseAttentionSpecs]] = None,
    **kwargs,
) -> dict:
    """
    Estimate per-GPU VRAM for a model without a Ray cluster.

    max_model_len = max_input_prompt_length + max_output_prompt_length.
    max_num_batched_tokens defaults to max_num_seqs.
    """
    max_model_len = max_input_prompt_length + max_output_prompt_length
    if max_num_batched_tokens is None:
        max_num_batched_tokens = max_num_seqs

    hf_params = normalize_hf_config(load_hf_config_dict(model_name))
    vram_reqs = build_vram_reqs(
        hf_params,
        attention_cls=attention_cls,
        tensor_parallel_size=tensor_parallel_size,
        **kwargs,
    )

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
        "weights_gb": weights_gb,
        "kv_cache_gb": kv_cache_gb,
        "activation_gb": activation_gb,
        "overhead_gb": overhead_gb,
        "misc_gb": misc_gb,
        "total_vram_gb": total_vram_gb,
    }
