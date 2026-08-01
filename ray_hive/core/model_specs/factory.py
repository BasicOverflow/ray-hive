"""Detect multimodal / pooling model families for VRAM class selection."""
from typing import Optional, Type

from .attention import BaseAttentionSpecs
from .mm_attention import MultimodalAttentionSpecs
from .mm_vram_reqs import MultimodalVramReqs
from .vram_reqs import BaseVramReqs


def is_pooling_kwargs(vllm_kwargs: dict | None) -> bool:
    """True when deploy targets pooling/embed runner via vllm_kwargs."""
    kw = vllm_kwargs or {}
    return kw.get("task") == "embed" or kw.get("runner") == "pooling"


def is_multimodal_hf(hf_params: dict) -> bool:
    """True when HF config exposes vision_config or audio_config."""
    return (
        isinstance(hf_params.get("vision_config"), dict)
        or isinstance(hf_params.get("audio_config"), dict)
    )


def default_limit_mm_per_prompt(hf_params: dict) -> dict:
    """Default limit_mm_per_prompt from HF modality configs when user omitted it."""
    limit: dict[str, int] = {}
    if isinstance(hf_params.get("vision_config"), dict):
        limit["image"] = 1
    if isinstance(hf_params.get("audio_config"), dict):
        limit["audio"] = 1
    return limit


def resolve_limit_mm_per_prompt(hf_params: dict, vllm_kwargs: dict | None) -> dict:
    """Return limit_mm_per_prompt from kwargs or MM default."""
    kw = vllm_kwargs or {}
    if "limit_mm_per_prompt" in kw and kw["limit_mm_per_prompt"] is not None:
        return dict(kw["limit_mm_per_prompt"])
    return default_limit_mm_per_prompt(hf_params)


def select_vram_classes(
    hf_params: dict,
    attention_cls: Optional[Type[BaseAttentionSpecs]] = None,
) -> tuple[Type[BaseAttentionSpecs], Type[BaseVramReqs]]:
    """Pick attention + VramReqs classes (multimodal vs text). Pooling is a flag."""
    multimodal = is_multimodal_hf(hf_params)
    if multimodal:
        return attention_cls or MultimodalAttentionSpecs, MultimodalVramReqs
    return attention_cls or BaseAttentionSpecs, BaseVramReqs
