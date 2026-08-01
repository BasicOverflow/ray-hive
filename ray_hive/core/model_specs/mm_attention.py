"""Multimodal attention specs — LM KV plus MM placeholder token budgets."""
from typing import Any

from ray_hive.core.ray_utils.mm_helpers import (
    audio_tokens_per_item,
    mm_count,
    vision_tokens_per_image,
    vision_tokens_per_video,
)

from .attention import BaseAttentionSpecs


class MultimodalAttentionSpecs(BaseAttentionSpecs):
    """
    KV calculator for generative multimodal models.

    LM KV uses flattened text_config fields (via BaseAttentionSpecs).
    mm_tokens_per_prompt sizes worst-case image/video/audio placeholders from
    limit_mm_per_prompt and HF vision/audio configs.
    """

    def __init__(
        self,
        kv_bytes_per_element: float | None = None,
        tensor_parallel_size: int = 1,
        limit_mm_per_prompt: dict | None = None,
        mm_processor_kwargs: dict | None = None,
        media_io_kwargs: dict | None = None,
        **hf_params: Any,
    ):
        super().__init__(
            kv_bytes_per_element=kv_bytes_per_element,
            tensor_parallel_size=tensor_parallel_size,
            **hf_params,
        )
        self.limit_mm_per_prompt = dict(limit_mm_per_prompt or {})
        self.mm_processor_kwargs = dict(mm_processor_kwargs or {})
        self.media_io_kwargs = dict(media_io_kwargs or {})


    def mm_tokens_per_prompt(
        self,
        limit_mm_per_prompt: dict | None = None,
        mm_processor_kwargs: dict | None = None,
        media_io_kwargs: dict | None = None,
    ) -> int:
        """Worst-case MM placeholder tokens for one prompt at the given limits."""
        limit = limit_mm_per_prompt if limit_mm_per_prompt is not None else self.limit_mm_per_prompt
        proc = mm_processor_kwargs if mm_processor_kwargs is not None else self.mm_processor_kwargs
        _ = media_io_kwargs if media_io_kwargs is not None else self.media_io_kwargs

        total = 0
        n_img = mm_count(limit, "image")
        if n_img:
            total += n_img * vision_tokens_per_image(self.hf_params, limit, proc)
        n_vid = mm_count(limit, "video")
        if n_vid:
            total += n_vid * vision_tokens_per_video(self.hf_params, limit, proc)
        n_aud = mm_count(limit, "audio")
        if n_aud:
            total += n_aud * audio_tokens_per_item(self.hf_params, limit)
        return total


    def max_tokens_per_mm_item(
        self,
        limit_mm_per_prompt: dict | None = None,
        mm_processor_kwargs: dict | None = None,
    ) -> int:
        """
        Max tokens for one MM item across supported modalities.

        vLLM validates max_num_batched_tokens against this even when a modality
        is limited to 0 at request time (prefix-LM / disable_chunked_mm_input).
        """
        limit = limit_mm_per_prompt if limit_mm_per_prompt is not None else self.limit_mm_per_prompt
        proc = mm_processor_kwargs if mm_processor_kwargs is not None else self.mm_processor_kwargs
        sizes: list[int] = []
        if isinstance(self.hf_params.get("vision_config"), dict):
            sizes.append(vision_tokens_per_image(self.hf_params, limit, proc))
            sizes.append(vision_tokens_per_video(self.hf_params, limit, proc))
        if isinstance(self.hf_params.get("audio_config"), dict):
            sizes.append(audio_tokens_per_item(self.hf_params, limit))
        return max(sizes) if sizes else 0


    def effective_input_len(
        self,
        text_input_len: int,
        limit_mm_per_prompt: dict | None = None,
        mm_processor_kwargs: dict | None = None,
        media_io_kwargs: dict | None = None,
    ) -> int:
        """Text input tokens plus worst-case MM placeholder tokens."""
        return text_input_len + self.mm_tokens_per_prompt(
            limit_mm_per_prompt, mm_processor_kwargs, media_io_kwargs
        )
