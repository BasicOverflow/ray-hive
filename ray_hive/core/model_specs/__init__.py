"""Public API for model VRAM specs and deployment planning."""
from .attention import BaseAttentionSpecs, TensorParallelAttentionSpecs
from .planner import build_vram_reqs, plan_deployment
from .vram_reqs import BaseVramReqs, Qwen35VramReqs

__all__ = [
    "BaseAttentionSpecs",
    "TensorParallelAttentionSpecs",
    "BaseVramReqs",
    "Qwen35VramReqs",
    "build_vram_reqs",
    "plan_deployment",
]
