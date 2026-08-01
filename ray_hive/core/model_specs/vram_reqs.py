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

from typing import Any, Optional, Type

from .attention import BaseAttentionSpecs

# vLLM CUDA-graph capture outside the util pool (~1–3 GiB).
CUDA_GRAPH_HEADROOM_GB = 2.0
# FlashInfer logits + masked + softmax scratch outside the util pool.
SAMPLER_BYTES_PER_SEQ_VOCAB = 10.0


class BaseVramReqs:
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
        self.tp_size = int(tensor_parallel_size)
        self.pooling = False

        # Default to standard transformer attention when no custom class is provided.
        # kv_bytes_per_element=None → AttentionSpecs resolves from kv_cache_dtype.
        cls = attention_cls or BaseAttentionSpecs
        self.attention = cls(
            kv_bytes_per_element=kv_cache_dtype_bytes,
            tensor_parallel_size=self.tp_size,
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
            "int4": 0.5, "uint4": 0.5, "nf4": 0.5, "fp4": 0.5,
        }
        for key in keys:
            value = self.hf_params.get(key)
            if value is None:
                continue
            name = str(value).lower().split(".")[-1]
            if name in dtype_sizes:
                return dtype_sizes[name]
        return default


    def _bits_to_bytes(self, bits: float) -> float:
        return float(bits) / 8.0


    def _quant_weight_bits(self, quant: dict) -> float | None:
        """Weight bit-width from HF quantization_config, or None if unknown."""
        for key in ("bits", "weight_bits", "w_bit", "wbits", "num_bits"):
            val = quant.get(key)
            if val is not None:
                return float(val)
        if quant.get("load_in_4bit"):
            return 4.0
        if quant.get("load_in_8bit"):
            return 8.0

        group_bits: list[float] = []
        for group in (quant.get("config_groups") or {}).values():
            if not isinstance(group, dict):
                continue
            w = group.get("weights")
            if isinstance(w, dict) and w.get("num_bits") is not None:
                group_bits.append(float(w["num_bits"]))
        if group_bits:
            # Prefer the most common Linear scheme; if mixed, take max bits
            # so weight VRAM is not under-estimated.
            return max(group_bits)

        blob = " ".join(
            str(quant.get(k) or "")
            for k in ("quant_method", "format", "fmt", "quant_type", "bnb_4bit_quant_type")
        ).lower()
        if any(s in blob for s in ("fp8", "float8", "float-quantized")):
            return 8.0
        if any(s in blob for s in ("nf4", "fp4", "int4", "w4a", "pack-quantized", "pack_quantized")):
            return 4.0
        if any(s in blob for s in ("int8", "w8a")):
            return 8.0
        method = str(quant.get("quant_method") or "").lower()
        # AWQ/GPTQ checkpoints almost always omit bits only when 4-bit.
        if method in ("awq", "gptq", "squeezellm", "marlin", "gguf", "quark"):
            return 4.0
        return None


    def _quant_activation_bits(self, quant: dict) -> float | None:
        """Activation bit-width when the scheme quantizes activations; else None."""
        group_bits: list[float] = []
        for group in (quant.get("config_groups") or {}).values():
            if not isinstance(group, dict):
                continue
            a = group.get("input_activations")
            if isinstance(a, dict) and a.get("num_bits") is not None:
                group_bits.append(float(a["num_bits"]))
        if group_bits:
            return max(group_bits)
        blob = " ".join(
            str(quant.get(k) or "")
            for k in ("quant_method", "format", "fmt")
        ).lower()
        # Weight-only schemes (AWQ/GPTQ/W4A16) keep bf16/fp16 activations.
        if any(s in blob for s in ("awq", "gptq", "squeezellm", "marlin", "pack-quantized", "pack_quantized")):
            return None
        if any(s in blob for s in ("fp8", "float8", "float-quantized")):
            return 8.0
        if "w8a8" in blob or "w4a8" in blob:
            return 8.0
        return None


    def _param_dtype_bytes(self) -> float:
        """Return model *weight* dtype size in bytes (quant-aware)."""
        quant = self.hf_params.get("quantization_config")
        if isinstance(quant, dict):
            bits = self._quant_weight_bits(quant)
            if bits is not None:
                return self._bits_to_bytes(bits)
        for key in ("quantization",):
            val = self.hf_params.get(key)
            if val is None:
                continue
            s = str(val).lower()
            if "fp8" in s or "float8" in s:
                return 1.0
            if any(x in s for x in ("awq", "gptq", "marlin", "nf4", "fp4", "int4", "w4a")):
                return 0.5
            if "int8" in s or "w8a" in s:
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
        """Return activation element size (may differ from weight-only quants)."""
        quant = self.hf_params.get("quantization_config")
        if isinstance(quant, dict):
            bits = self._quant_activation_bits(quant)
            if bits is not None:
                return self._bits_to_bytes(bits)
        for key in ("quantization",):
            val = self.hf_params.get(key)
            if val is None:
                continue
            s = str(val).lower()
            if "fp8" in s or "float8" in s or "w8a8" in s or "w4a8" in s:
                return 1.0
        # Weight-only: activations stay at model compute dtype.
        explicit = self.hf_params.get("dtype")
        if explicit is not None and str(explicit).lower() not in ("auto",):
            return self._dtype_bytes("dtype", default=2)
        torch_dtype = str(self.hf_params.get("torch_dtype", "")).lower().split(".")[-1]
        if torch_dtype in ("float32", "fp32", "float"):
            return 2.0
        return self._dtype_bytes("torch_dtype", "dtype", default=2)

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

        Gemma4-style extras:
        - Per-Layer Embeddings (PLE): vocab × layers × ple_dim
        - use_double_wide_mlp on KV-shared layers: gated FFN at 2× intermediate
        - dual attention: full layers may use global_head_dim
        """
        hidden = self._hf("hidden_size")
        vocab = self._hf("vocab_size")
        intermediate = self._hf("intermediate_size")
        layers = self._hf("num_hidden_layers")

        kv_heads = self.attention.kv_heads
        head_dim = self.attention.head_dim
        global_head_dim = int(self.hf_params.get("global_head_dim") or head_dim)
        n_q_heads = int(
            self.hf_params.get("num_attention_heads")
            or self.hf_params.get("num_heads")
            or max(1, hidden // max(head_dim, 1))
        )
        layer_types = self.hf_params.get("layer_types")
        if not isinstance(layer_types, list) or len(layer_types) != layers:
            layer_types = ["full_attention"] * layers


        def _attn_params(layer_type: str) -> float:
            hd = global_head_dim if layer_type == "full_attention" else head_dim
            q_width = n_q_heads * hd
            kv_width = kv_heads * hd
            return float(2 * hidden * q_width + 2 * hidden * kv_width)


        act = str(
            self.hf_params.get("mlp_hidden_act")
            or self.hf_params.get("hidden_act")
            or self.hf_params.get("hidden_activation", "silu")
        ).lower()
        # Gemma4 double-wide MLP uses fused gate+up (GEGLU-style) even when
        # hidden_activation is gelu_*; plain gelu 2-mat undercounts badly.
        gated = bool(self.hf_params.get("use_double_wide_mlp")) or act not in (
            "gelu", "gelu_new", "gelu_pytorch_tanh", "gelu_fast", "relu", "relu2", "tanh",
        )
        if gated:
            mlp = 3 * hidden * intermediate
        else:
            mlp = 2 * hidden * intermediate

        embed_factor = 1 if self.hf_params.get("tie_word_embeddings") else 2
        embed = embed_factor * vocab * hidden

        # Per-layer embeddings (Gemma 4 E2B/E4B): large lookup tables, still loaded in VRAM.
        ple_dim = int(self.hf_params.get("hidden_size_per_layer_input") or 0)
        ple_vocab = int(self.hf_params.get("vocab_size_per_layer_input") or 0)
        ple = 0.0
        if ple_dim > 0 and ple_vocab > 0:
            ple = float(ple_vocab * layers * ple_dim)
            # context-aware projection: hidden → layers * ple_dim
            ple += float(hidden * layers * ple_dim)

        double_wide = bool(self.hf_params.get("use_double_wide_mlp"))
        n_shared = int(self.hf_params.get("num_kv_shared_layers") or 0) if double_wide else 0
        # KV-shared layers: 2× intermediate with gated FFN.
        mlp_wide = 3 * hidden * (2 * intermediate) if double_wide else mlp

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
            attn = _attn_params("full_attention")
            params = embed + ple
            for ch in pattern:
                if ch == "*":
                    params += attn
                elif ch == "-":
                    params += mlp
                elif ch == "M":
                    params += mamba
        else:
            params = embed + ple
            shared_start = layers - n_shared if n_shared > 0 else layers
            for i in range(layers):
                params += _attn_params(layer_types[i])
                params += mlp_wide if i >= shared_start else mlp

        bytes_per_param = self._param_dtype_bytes()
        return (params * bytes_per_param) / (1024 ** 3) / self.tp_size

    def calc_misc_vram_gb(self) -> float:
        """
        Default non-KV runtime overhead.
        Can be overridden for hybrid models.
        """
        misc = 0.0

        if self.hf_params.get("num_experts"):
            hidden = self._hf("hidden_size")
            experts = self.hf_params.get("num_experts_per_tok")
            if experts is None:
                experts = self.hf_params.get("top_k_experts")
            if experts:
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
        # AttentionSpecs already reports per-GPU KV bytes (tp_size applied there).
        bytes_total = (
            self.attention.kv_bytes_per_sequence(max_model_len)
            * max_num_seqs
        )
        return bytes_total / (1024 ** 3)

    # ------------------------------------------------------------
    # TOTAL MEMORY
    # ------------------------------------------------------------

    def calc_sleep_peak_gb(self, sleep_mode: bool = False) -> float:
        """Extra profiled peak under vLLM enable_sleep_mode (~weights again)."""
        return self.calc_weights_gb() if sleep_mode else 0.0


    def calc_cuda_graph_gb(
        self,
        enforce_eager: bool = False,
        usable_gb: float | None = None,
        min_pool_gb: float | None = None,
    ) -> float:
        """Outside-util CUDA-graph headroom (0 when eager or it would starve KV)."""
        if enforce_eager:
            return 0.0
        g = CUDA_GRAPH_HEADROOM_GB
        if (
            usable_gb is not None
            and min_pool_gb is not None
            and usable_gb - g < min_pool_gb
            and usable_gb >= min_pool_gb
        ):
            return 0.0
        return g


    def calc_sampler_scratch_gb(self, max_num_seqs: int) -> float:
        """Outside-util generate sampler scratch; 0 for pooling."""
        if self.pooling or max_num_seqs <= 0:
            return 0.0
        vocab = float(self.hf_params["vocab_size"])
        return max_num_seqs * vocab * SAMPLER_BYTES_PER_SEQ_VOCAB / (1024 ** 3)


    def sampler_bytes_per_seq(self) -> float:
        """Per-seq sampler scratch bytes (0 when pooling)."""
        if self.pooling:
            return 0.0
        return float(self.hf_params["vocab_size"]) * SAMPLER_BYTES_PER_SEQ_VOCAB


    def calc_fixed_non_kv_gb(self, sleep_mode: bool = False) -> float:
        """Weights + overhead + misc + sleep peak (before activations/KV)."""
        return (
            self.calc_system_overhead_gb()
            + self.calc_weights_gb()
            + self.calc_misc_vram_gb()
            + self.calc_sleep_peak_gb(sleep_mode)
        )


    def calc_non_kv_vram_gb(
        self,
        max_num_batched_tokens: int,
        sleep_mode: bool = False,
    ) -> float:
        """Return non-KV VRAM (fixed + activations) in GiB."""
        return (
            self.calc_fixed_non_kv_gb(sleep_mode)
            + self.calc_activation_gb(max_num_batched_tokens)
        )
