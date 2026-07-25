"""
VRAM requirement estimator for transformer inference systems.

This module separates VRAM into:
- system overhead (CUDA, NCCL, driver)
- model weights
- KV cache (via AttentionSpecs)
- activations (depends on batched tokens)
- model-specific runtime overhead (MoE, hybrid models, etc.)

Designed to work with vLLM-style inference systems where KV cache is dynamically
allocated from a pre-reserved memory pool.
"""

from abc import ABC
from typing import Any, Optional, Type

from .attention import BaseAttentionSpecs


class BaseVramReqs(ABC):
    """
    Base calculator for model VRAM requirements.
    """

    def __init__(
        self,
        kv_cache_dtype_bytes: float | None = None,
        attention_cls: Optional[Type[BaseAttentionSpecs]] = None,
        tensor_parallel_size: int = 1,
        **hf_params: Any,
    ):
        """Store HF config and build the attention specs calculator."""
        self.hf_params = hf_params
        self.tp_size = max(1, int(tensor_parallel_size))

        if kv_cache_dtype_bytes is None:
            kv_cache_dtype_bytes = 1

        # Default to standard transformer attention when no custom class is provided.
        cls = attention_cls or BaseAttentionSpecs
        self.attention = cls(
            kv_bytes_per_element=kv_cache_dtype_bytes,
            **hf_params
        )

    def _hf(self, name: str):
        """Return a Hugging Face config value by name."""
        return self.hf_params[name]


    def _dtype_bytes(self, *keys: str, default: float = 4) -> float:
        """Return element size in bytes from HF config dtype fields."""
        dtype_sizes = {
            "float64": 8, "fp64": 8, "double": 8,
            "float32": 4, "fp32": 4, "float": 4,
            "bfloat16": 2, "bf16": 2,
            "float16": 2, "fp16": 2, "half": 2,
            "float8": 1, "fp8": 1,
            "int8": 1, "uint8": 1,
        }
        for key in keys:
            value = self.hf_params.get(key)
            if value is None:
                continue
            name = str(value).lower().split(".")[-1]
            if name in dtype_sizes:
                return dtype_sizes[name]
        return default


    def _param_dtype_bytes(self) -> float:
        """Return model parameter dtype size in bytes."""
        # FP8 HF checkpoints often keep torch_dtype=bfloat16 while weights are fp8.
        quant = self.hf_params.get("quantization_config")
        if isinstance(quant, dict):
            for key in ("quant_method", "fmt", "bits"):
                val = quant.get(key)
                if val is not None and ("fp8" in str(val).lower() or "float8" in str(val).lower()):
                    return 1.0
        for key in ("quantization",):
            val = self.hf_params.get(key)
            if val is not None and ("fp8" in str(val).lower() or "float8" in str(val).lower()):
                return 1.0
        # Explicit vLLM dtype= wins over HF torch_dtype.
        explicit = self.hf_params.get("dtype")
        if explicit is not None and str(explicit).lower() not in ("auto",):
            return self._dtype_bytes("dtype", default=2)
        # HF often labels float32 while vLLM dtype=auto loads fp16/bf16 for inference.
        torch_dtype = str(self.hf_params.get("torch_dtype", "")).lower().split(".")[-1]
        if torch_dtype in ("float32", "fp32", "float"):
            return 2.0
        return self._dtype_bytes("torch_dtype", "dtype", default=2)


    def _activation_dtype_bytes(self) -> float:
        """Return activation dtype size in bytes."""
        # Override if runtime kernels use activation precision different from model weights.
        return self._param_dtype_bytes()

    # ------------------------------------------------------------
    # CORE VRAM COMPONENTS
    # ------------------------------------------------------------

    def calc_system_overhead_gb(self) -> float:
        """
        Default runtime overhead (CUDA, driver, NCCL, vLLM runtime).

        When tensor_parallel_size > 1, add a coarse per-GPU NCCL/TP allowance
        (not an exact NCCL formula — leave headroom in the deploy budget too).
        """
        base = 0.25
        if self.tp_size > 1:
            base += 0.5
        return base

    def calc_weights_gb(self) -> float:
        """
        HF-derived weight estimate (per GPU when TP > 1).

        Dense transformers: every layer is attn + MLP.
        hybrid_override_pattern (NemotronH): per-char layer type —
        '*' attention, '-' MLP, 'M' Mamba-2 — so we do not count attn+MLP on every layer.
        """
        hidden = self._hf("hidden_size")
        vocab = self._hf("vocab_size")
        intermediate = self._hf("intermediate_size")

        kv_width = self.attention.kv_heads * self.attention.head_dim
        attn = (2 * hidden * hidden) + (2 * hidden * kv_width)

        act = str(
            self.hf_params.get("mlp_hidden_act")
            or self.hf_params.get("hidden_act", "silu")
        ).lower()
        if act in ("gelu", "gelu_new", "gelu_pytorch_tanh", "gelu_fast", "relu", "relu2", "tanh"):
            mlp = 2 * hidden * intermediate
        else:
            mlp = 3 * hidden * intermediate

        embed_factor = 1 if self.hf_params.get("tie_word_embeddings") else 2
        embed = embed_factor * vocab * hidden

        pattern = self.hf_params.get("hybrid_override_pattern")
        if isinstance(pattern, str) and pattern:
            d_inner = int(self.hf_params.get("mamba_num_heads", 0)) * int(
                self.hf_params.get("mamba_head_dim", 0)
            )
            if d_inner <= 0:
                d_inner = int(self.hf_params.get("expand", 2)) * hidden
            d_conv = int(self.hf_params.get("conv_kernel", 4))
            # in_proj (~2*d_inner) + out_proj + depthwise conv — coarse Mamba-2 block size
            mamba = 3 * hidden * d_inner + d_inner * d_conv
            params = embed
            for ch in pattern:
                if ch == "*":
                    params += attn
                elif ch == "-":
                    params += mlp
                elif ch == "M":
                    params += mamba
        else:
            layers = self._hf("num_hidden_layers")
            params = layers * (attn + mlp) + embed

        bytes_per_param = self._param_dtype_bytes()
        return (params * bytes_per_param) / (1024 ** 3) / self.tp_size

    def calc_misc_vram_gb(self) -> float:
        """
        Default non-KV runtime overhead.
        Can be overridden for hybrid models.
        """
        misc = 0.0

        if "num_experts" in self.hf_params:
            hidden = self._hf("hidden_size")
            experts = self._hf("num_experts_per_tok")
            misc += experts * hidden * 2 / (1024 ** 3) * 0.01

        return misc

    # ------------------------------------------------------------
    # ACTIVATIONS
    # ------------------------------------------------------------

    def calc_activation_gb(self, max_num_batched_tokens: int) -> float:
        """
        Activation memory.

        IMPORTANT:
        - Uses activation dtype, not KV dtype
        - Still a kernel-level peak memory approximation
        - Override for FlashAttention/eager differences, fused kernels, or checkpointing
        """

        hidden_size = self._hf("hidden_size")
        bytes_per_element = self._activation_dtype_bytes()

        # Represents fused QKV + MLP + residual buffers in vLLM-style kernels.
        activation_peak_multiplier = 1.5

        bytes_total = (
            max_num_batched_tokens
            * hidden_size
            * bytes_per_element
            * activation_peak_multiplier
        )

        return bytes_total / (1024 ** 3)

    # ------------------------------------------------------------
    # KV CACHE
    # ------------------------------------------------------------

    def calc_kv_cache_gb(self, max_model_len: int, max_num_seqs: int) -> float:
        """Return per-GPU KV cache VRAM in GiB for max_model_len and max_num_seqs."""
        # Correctness for hybrid, sliding-window, or partial-KV models belongs in AttentionSpecs.
        bytes_total = (
            self.attention.kv_bytes_per_sequence(max_model_len)
            * max_num_seqs
        )

        return bytes_total / (1024 ** 3) / self.tp_size

    # ------------------------------------------------------------
    # TOTAL MEMORY
    # ------------------------------------------------------------

    def calc_non_kv_vram_gb(self, max_num_batched_tokens: int) -> float:
        """Return non-KV VRAM (overhead + weights + misc + activations) in GiB."""
        return (
            self.calc_system_overhead_gb()
            + self.calc_weights_gb()
            + self.calc_misc_vram_gb()
            + self.calc_activation_gb(max_num_batched_tokens)
        )
