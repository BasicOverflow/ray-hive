"""Public API for model VRAM specs and deployment planning."""
from .attention import BaseAttentionSpecs
from .estimate import estimate_vram, load_hf_config_dict
from .factory import is_multimodal_hf, is_pooling_kwargs, select_vram_classes
from .mm_attention import MultimodalAttentionSpecs
from .mm_vram_reqs import MultimodalVramReqs
from .planner import build_vram_reqs, plan_deployment
from .vram_reqs import BaseVramReqs

__all__ = [
    "BaseAttentionSpecs",
    "BaseVramReqs",
    "MultimodalAttentionSpecs",
    "MultimodalVramReqs",
    "build_vram_reqs",
    "estimate_vram",
    "is_multimodal_hf",
    "is_pooling_kwargs",
    "load_hf_config_dict",
    "plan_deployment",
    "select_vram_classes",
]
