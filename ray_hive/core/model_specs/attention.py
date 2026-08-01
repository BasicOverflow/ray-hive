"""
Attention specs for KV-cache sizing and finding optimal max concurrent sequences.

These classes read Hugging Face config fields and compute KV bytes per token,
bytes per sequence, and max sequences for a given KV budget. BaseAttentionSpecs
covers standard transformer attention; subclasses override KV heads, KV layers,
or sequence length. Downstream VRAM calculators use them after weights, overhead,
and other non-KV memory are accounted for.

Tensor-parallel per-GPU KV sharding is applied here (tp_size). Weight sharding
stays in VramReqs.

NOTE: Standard KV formula will always be upper bound compared to more efficient attentions,
so if custom KV calculation not provided for given model, it will default to standard KV formula.
"""
from typing import Any
import math


def resolve_kv_bytes_per_element(hf_params: dict, explicit: float | None = None) -> float:
    """
    KV element size from an explicit override, else kv_cache_dtype / cache_dtype.

    auto / unset → 2 (bf16/fp16). fp8 / float8 / int8 → 1. int4 / fp4 / nf4 → 0.5.
    """
    if explicit is not None:
        return float(explicit)
    dtype = str(
        hf_params.get("kv_cache_dtype")
        or hf_params.get("cache_dtype")
        or "auto"
    ).lower()
    if "fp8" in dtype or "float8" in dtype or dtype in ("int8", "uint8"):
        return 1.0
    if any(s in dtype for s in ("int4", "uint4", "fp4", "nf4")):
        return 0.5
    if dtype in ("float32", "fp32", "float"):
        return 4.0
    return 2.0


class BaseAttentionSpecs:
    """
    Base KV-cache calculator for transformer-style attention.

    Args:
        kv_bytes_per_element:
            Size of each K/V cache element in bytes. None → resolve from
            hf_params kv_cache_dtype (auto/bf16=2, fp8=1, …).

        tensor_parallel_size:
            TP world size; KV bytes are reported per GPU.

        **hf_params:
            HuggingFace config parameters.
    """

    def __init__(
        self,
        kv_bytes_per_element: float | None = None,
        tensor_parallel_size: int = 1,
        **hf_params: Any,
    ):
        """Store Hugging Face config params used by cache formulas."""
        self.hf_params = hf_params
        self.kv_bytes_per_element = resolve_kv_bytes_per_element(
            hf_params, kv_bytes_per_element
        )
        self.tp_size = max(1, int(tensor_parallel_size))


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
        """Return per-GPU KV cache bytes needed per token."""
        full = 2 * self.kv_bytes_per_element * self.kv_layers * self.kv_heads * self.head_dim
        return full / self.tp_size


    def kv_bytes_per_sequence(self, max_model_len: int) -> float:
        """Return per-GPU KV cache bytes needed for one sequence."""
        assert max_model_len > 0, f"max_model_len must be positive, got {max_model_len}"
        return self.kv_bytes_per_token() * max_model_len


    def calc_max_num_seqs_given_kv_cache(self, max_model_len: int, kv_cache_gib: float) -> int:
        """Find max concurrent sequences given per-GPU KV budget and max_model_len."""
        assert kv_cache_gib > 0, f"kv_cache_gib must be positive, got {kv_cache_gib}"
        kv_budget_bytes = kv_cache_gib * (1024 ** 3)
        bytes_per_seq = self.kv_bytes_per_sequence(max_model_len)
        return max(1, math.floor(kv_budget_bytes / bytes_per_seq))


    def mm_tokens_per_prompt(self) -> int:
        """Worst-case MM placeholder tokens (0 for text-only)."""
        return 0


    def max_tokens_per_mm_item(self) -> int:
        """Largest single MM item for encoder/BT floor (0 for text-only)."""
        return 0


    def effective_input_len(self, text_input_len: int) -> int:
        """Text input length plus MM placeholders when applicable."""
        return text_input_len
