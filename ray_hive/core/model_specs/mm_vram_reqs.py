"""VRAM requirements for generative multimodal models."""
from typing import Any, Optional, Type

from ray_hive.core.ray_utils.mm_helpers import estimate_encoder_params

from .attention import BaseAttentionSpecs
from .mm_attention import MultimodalAttentionSpecs
from .vram_reqs import BaseVramReqs


class MultimodalVramReqs(BaseVramReqs):
    """
    VRAM calculator for generative VL/AV models.

    Adds encoder/projector weights and a coarse encoder-cache term on top of
    BaseVramReqs LM math. Encoder TP follows mm_encoder_tp_mode:
    - "weights" (default): shard encoder+projector across TP ranks
    - "data": replicate encoder+projector on each rank
    """

    def __init__(
        self,
        kv_cache_dtype_bytes: float | None = None,
        attention_cls: Optional[Type[BaseAttentionSpecs]] = None,
        tensor_parallel_size: int = 1,
        limit_mm_per_prompt: dict | None = None,
        mm_processor_kwargs: dict | None = None,
        media_io_kwargs: dict | None = None,
        mm_encoder_tp_mode: str = "weights",
        mm_processor_cache_gb: float | None = None,
        **hf_params: Any,
    ):
        self.limit_mm_per_prompt = dict(limit_mm_per_prompt or {"image": 1})
        self.mm_processor_kwargs = dict(mm_processor_kwargs or {})
        self.media_io_kwargs = dict(media_io_kwargs or {})
        self.mm_encoder_tp_mode = str(mm_encoder_tp_mode or "weights")
        self.mm_processor_cache_gb = float(mm_processor_cache_gb or 0)

        attn = attention_cls or MultimodalAttentionSpecs
        # Strip MM-only keys from hf_params so BaseAttentionSpecs subclasses
        # that only accept known kwargs still work when overridden.
        super().__init__(
            kv_cache_dtype_bytes=kv_cache_dtype_bytes,
            attention_cls=attn,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            mm_processor_kwargs=self.mm_processor_kwargs,
            media_io_kwargs=self.media_io_kwargs,
            **hf_params,
        )


    def calc_lm_weights_gb(self) -> float:
        """Language-tower weights only (per GPU)."""
        return super().calc_weights_gb()


    def calc_mm_encoder_weights_gb(self) -> float:
        """Vision/audio encoder + projector weights (per GPU after TP policy)."""
        bytes_per = self._param_dtype_bytes()
        params = 0.0

        vision = self.hf_params.get("vision_config")
        if isinstance(vision, dict):
            params += estimate_encoder_params(vision)

        audio = self.hf_params.get("audio_config")
        if isinstance(audio, dict):
            params += estimate_encoder_params(audio, default_hidden=1280)

        # Projector / merger into LM hidden (vision + audio when both present).
        lm_hidden = int(self.hf_params.get("hidden_size") or 0)
        if lm_hidden and params > 0:
            if isinstance(vision, dict):
                vis_hidden = int(vision.get("hidden_size") or vision.get("width") or 0)
                merge = max(1, int(vision.get("spatial_merge_size") or 1))
                out_h = int(vision.get("out_hidden_size") or lm_hidden)
                if vis_hidden:
                    if merge > 1:
                        ctx = vis_hidden * merge * merge
                        params += ctx * ctx + ctx * out_h
                    else:
                        params += 2.0 * vis_hidden * lm_hidden
            if isinstance(audio, dict):
                aud_hidden = int(audio.get("hidden_size") or audio.get("d_model") or 0)
                aud_out = int(audio.get("output_proj_dims") or lm_hidden)
                # estimate_encoder_params already counts output_proj_dims once; add
                # embed_audio-style second projection when dims differ from a plain linear.
                if aud_hidden and not audio.get("output_proj_dims"):
                    params += float(aud_hidden * aud_out)

        gb = (params * bytes_per) / (1024 ** 3)
        if self.mm_encoder_tp_mode == "data":
            return gb
        return gb / self.tp_size


    def calc_weights_gb(self) -> float:
        """LM weights + multimodal encoder/projector weights (per GPU)."""
        return self.calc_lm_weights_gb() + self.calc_mm_encoder_weights_gb()


    def calc_system_overhead_gb(self) -> float:
        """Text overhead plus encoder-forward / allocator peak seen in vLLM profiles."""
        base = super().calc_system_overhead_gb()
        enc = self.calc_mm_encoder_weights_gb()
        # Vision+audio towers (Gemma 4 E2B) need more peak than weight bytes alone.
        towers = 0
        if isinstance(self.hf_params.get("vision_config"), dict):
            towers += 1
        if isinstance(self.hf_params.get("audio_config"), dict):
            towers += 1
        return base + max(0.75 * max(towers, 1), 0.75 * enc)


    def calc_encoder_cache_gb(self) -> float:
        """Coarse encoder-cache / MM activation reservation for planned MM limits."""
        mm_tokens = int(self.attention.mm_tokens_per_prompt())
        if mm_tokens <= 0:
            return 0.0

        vision = self.hf_params.get("vision_config")
        merge = 1
        enc_hidden = int(self.hf_params.get("hidden_size") or 2048)
        if isinstance(vision, dict):
            merge = max(1, int(vision.get("spatial_merge_size") or 1))
            enc_hidden = int(vision.get("hidden_size") or vision.get("width") or enc_hidden)
        # Pre-merge patch tokens are what the vision tower activates on.
        patch_tokens = mm_tokens * merge * merge
        bytes_per = self._activation_dtype_bytes()
        return (patch_tokens * enc_hidden * bytes_per * 2.0) / (1024 ** 3)


    def calc_misc_vram_gb(self) -> float:
        misc = super().calc_misc_vram_gb()
        misc += self.calc_encoder_cache_gb()
        misc += self.mm_processor_cache_gb
        return misc
