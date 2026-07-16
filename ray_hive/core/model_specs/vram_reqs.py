"""
VRAM requirement estimator for transformer inference systems.

This module separates VRAM into:
- system overhead (CUDA, NCCL, driver)
- model weights
- KV cache (via AttentionSpecs)
- activations (depends on batched tokens)
- model-specific runtime overhead (MoE, hybrid models, speculative decoding, etc.)

Designed to work with vLLM-style inference systems where KV cache is dynamically
allocated from a pre-reserved memory pool.
"""

from abc import ABC
from typing import Any

from .attention import BaseAttentionSpecs, Qwen35AttentionSpecs


class BaseVramReqs(ABC):
    """
    Base calculator for model VRAM requirements.
    """

    attention_cls: type[BaseAttentionSpecs] = BaseAttentionSpecs

    def __init__(self, speculative_decoding_enabled: bool = False, kv_cache_dtype_bytes: float | None = None, **hf_params: Any):
        """Store HF config and build the attention specs calculator."""
        self.hf_params = hf_params
        self.speculative_decoding_enabled = speculative_decoding_enabled

        if kv_cache_dtype_bytes is None:
            kv_cache_dtype_bytes = 1

        self.attention = self.attention_cls(
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
        # Override for mixed precision or selective quantization.
        return self._dtype_bytes("torch_dtype", "dtype")


    def _kv_dtype_bytes(self) -> float:
        """Return KV cache dtype size in bytes."""
        return self.attention.kv_bytes_per_element


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
        """
        return 0.5

    def calc_weights_gb(self) -> float:
        """
        HF-derived dense transformer weight estimate.
        Override for MoE, unusual projections, shared weights, or non-SwiGLU MLP blocks.
        """

        hidden = self._hf("hidden_size")
        layers = self._hf("num_hidden_layers")
        vocab = self._hf("vocab_size")
        intermediate = self._hf("intermediate_size")

        # Attention projections: Q + K + V + output.
        # Uses attention specs so GQA/MQA reduce K/V params while MHA stays 4 * hidden^2.
        kv_width = self.attention.kv_heads * self.attention.head_dim
        attn = (2 * hidden * hidden) + (2 * hidden * kv_width)

        # SwiGLU-style MLP (gate + up + down), common in modern decoder LLMs.
        # Override for GELU MLPs, fused/shared gate/up projections, or MoE blocks.
        mlp = 3 * hidden * intermediate

        # embeddings + optional untied LM head
        # Override for partial tying, output projection reuse, or learned positional embeddings.
        embed_factor = 1 if self.hf_params.get("tie_word_embeddings") else 2
        embed = embed_factor * vocab * hidden

        # Norms and biases are omitted here; override if those small systematic terms matter.
        params = layers * (attn + mlp) + embed

        bytes_per_param = self._param_dtype_bytes()
        return (params * bytes_per_param) / (1024 ** 3)

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
        """Return KV cache VRAM in GiB for max_model_len and max_num_seqs."""
        # Correctness for hybrid, sliding-window, or partial-KV models belongs in AttentionSpecs.
        bytes_total = (
            self.attention.kv_bytes_per_sequence(max_model_len)
            * max_num_seqs
        )

        return bytes_total / (1024 ** 3)

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

    def calc_total_gb(
        self,
        max_model_len: int,
        max_num_seqs: int,
        max_num_batched_tokens: int
    ) -> float:
        """Return total estimated VRAM in GiB for given deployment limits."""
        return (
            self.calc_non_kv_vram_gb(max_num_batched_tokens)
            + self.calc_kv_cache_gb(max_model_len, max_num_seqs)
        )


# ============================================================
# QWEN 3.5
# ============================================================

class Qwen35_SmallVarient_VramReqs(BaseVramReqs):
    """
    VRAM estimator for the Qwen3.5 Small (~0.6B) model.
    """

    attention_cls: type[BaseAttentionSpecs] = Qwen35AttentionSpecs

    # Published parameter count
    PARAM_COUNT = 610_000_000

    def calc_system_overhead_gb(self) -> float:
        """Return Qwen3.5 system overhead in GiB."""
        return 0.5

    def calc_weights_gb(self) -> float:
        """Return Qwen3.5 weight VRAM from published parameter count."""
        bytes_per_param = self._param_dtype_bytes()
        return (
            self.PARAM_COUNT * bytes_per_param
        ) / (1024 ** 3)

    def calc_misc_vram_gb(self) -> float:
        """Return Qwen3.5 kernel workspace and speculative decoding overhead."""
        hidden = self._hf("hidden_size")
        layers = self._hf("num_hidden_layers")

        # conservative kernel workspace estimate
        workspace = (
            hidden * layers * 2
        ) / (1024 ** 3) * 0.03

        misc = workspace

        # speculative decoding overhead
        if self.speculative_decoding_enabled:
            misc += self.calc_weights_gb() * 0.10

        misc += super().calc_misc_vram_gb()

        return misc


Qwen35VramReqs = Qwen35_SmallVarient_VramReqs