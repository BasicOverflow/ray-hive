"""
Ray-wired GPU allocators — bind abstract policies to Alive-node gating.

Hardware accessors live on BaseGpuAllocator (ray_utils.hardware); this module
only adds the Alive-node filter used on the live cluster.
"""
from ray_hive.core.gpu_alloc import (
    ConserveTdpAllocator,
    PerformanceAllocator,
    TensorParallelAllocator,
)
from ray_hive.core.ray_utils.hardware import filter_alive


class _AliveFilterMixin:
    """Drop GPUs whose registry host is not an Alive Ray node."""

    def filter_eligible(
        self,
        gpu_map: dict[str, dict],
        min_vram_gb: float,
        hf_params: dict,
        vllm_kwargs: dict,
    ) -> list[tuple[str, dict]]:
        return filter_alive(
            super().filter_eligible(gpu_map, min_vram_gb, hf_params, vllm_kwargs)
        )


class RayPerformanceAllocator(_AliveFilterMixin, PerformanceAllocator):
    """Performance policy with Alive-node gate."""


class RayConserveTdpAllocator(_AliveFilterMixin, ConserveTdpAllocator):
    """Conserve-TDP policy with Alive-node gate."""


class RayTensorParallelAllocator(_AliveFilterMixin, TensorParallelAllocator):
    """Same-node TP packing with Alive-node gate."""
