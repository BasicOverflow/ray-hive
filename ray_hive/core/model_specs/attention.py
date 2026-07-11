"""
Attention specs for KV-cache sizing and finding optimal max concurrent sequences.

These classes read Hugging Face config fields and compute KV bytes per token,
bytes per sequence, and max sequences for a given KV budget. BaseAttentionSpecs
covers standard transformer attention; subclasses override KV heads, KV layers,
sequence length, or tensor-parallel per-GPU sizing. Downstream VRAM calculators
use them after weights, overhead, and other non-KV memory are accounted for.

NOTE: Standard KV formula will always be upper bound compared to more efficient attentions, 
so if custom KV calculation not provided for given model, it will default to standard KV formula.
"""
from abc import ABC
from typing import Any
import math


class BaseAttentionSpecs(ABC):
    """
    Base KV-cache calculator for transformer-style attention.

    Args:
        kv_bytes_per_element:
            Size of each K/V cache element in bytes.

            Examples:
                FP64      = 8
                FP16/BF16 = 2
                FP8       = 1
                INT4      = 0.5

        **hf_params:
            HuggingFace config parameters.
    """

    def __init__(self, kv_bytes_per_element: float = 8, **hf_params: Any):
        """Store Hugging Face config params used by cache formulas."""
        self.hf_params = hf_params
        self.kv_bytes_per_element = kv_bytes_per_element


    def _hf(self, name: str):
        """Return a Hugging Face config value by name."""
        return self.hf_params[name]


    def _hf_any(self, *names: str):
        """Return the first present Hugging Face config value."""
        for name in names:
            value = self.hf_params.get(name)
            if value is not None:
                return value
        raise KeyError(f"Could not find any of: {', '.join(names)}")


    @property
    def head_dim(self) -> int:
        """Return the attention head dimension."""
        if self.hf_params.get("head_dim") is not None:
            return self.hf_params["head_dim"]

        hidden_size = self._hf_any(
            "hidden_size",
            "d_model"
        )

        num_heads = self._hf_any(
            "num_attention_heads",
            "num_heads",
            "n_head"
        )

        return hidden_size // num_heads


    @property
    def kv_heads(self) -> int:
        """Return the number of KV attention heads."""
        return self._hf_any(
            "num_key_value_heads",
            "num_kv_heads",
            "num_attention_heads",
            "num_heads",
            "n_head",
        )


    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers."""
        return self._hf_any(
            "num_hidden_layers",
            "num_layers",
            "n_layer",
            "n_layers"
        )


    @property
    def kv_layers(self) -> int:
        """Return the number of layers that contribute KV cache."""
        return self.num_layers


    def kv_bytes_per_token(self) -> float:
        """Return KV cache bytes needed per token."""
        return 2 * self.kv_bytes_per_element * self.kv_layers * self.kv_heads * self.head_dim


    def kv_bytes_per_sequence(self, max_model_len: int) -> float:
        """Return KV cache bytes needed for one sequence."""
        return self.kv_bytes_per_token() * max_model_len


    def calc_max_num_seqs_given_kv_cache(self, max_model_len: int, kv_cache_gib: float) -> int:
        """Find optimal max concurrent sequences given how much kv cache alloted to this deployment & max input+output prompt lengths."""
        kv_budget_bytes = kv_cache_gib * (1024 ** 3)
        bytes_per_seq = self.kv_bytes_per_sequence(max_model_len)
        return max(1, math.floor(kv_budget_bytes / bytes_per_seq))






# Example inheritance to calculate non-traditional attention.
class MQAAttentionSpecs(BaseAttentionSpecs):
    """KV cache calculator for multi-query attention models."""

    @property
    def kv_heads(self) -> int:
        """Return 1 — MQA uses a single shared KV head."""
        # MQA uses a single KV head shared across all query heads.
        return 1

    # NOTE: calc_max_num_seqs_given_kv_cache method stays the same, only difference is how kv_heads is calculated.
    # In Reality, Most MQA models expose:
    # {
    #   "num_attention_heads": 32,
    #   "num_key_value_heads": 1
    # }
    # In the hf config anyway, so the BaseAttentionSpecs class will already work for MQA models out of the box. However, this is just to demonstrate how to override a property from the base class.


# More custom attention inheritance examples:


class Qwen35AttentionSpecs(BaseAttentionSpecs):
    """KV cache calculator for Qwen3.x hybrid attention models."""

    @property
    def kv_layers(self) -> int:
        """Return Qwen hybrid layers that contribute KV cache."""
        return self.hf_params["num_attention_layers"]


# If later want per-GPU KV calculations for tensor parallel models
class TensorParallelAttentionSpecs(BaseAttentionSpecs):
    """KV cache calculator for per-GPU tensor parallel usage."""

    def __init__(self, tp_size: int, **kwargs):
        """Store tensor parallel size and attention config."""
        super().__init__(**kwargs)
        self.tp_size = tp_size


    def kv_bytes_per_token(self) -> float:
        """Return per-GPU KV cache bytes needed per token."""
        return super().kv_bytes_per_token() / self.tp_size



