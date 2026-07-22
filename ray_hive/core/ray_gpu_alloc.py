"""
Ray-wired GPU allocators — bind abstract policies to cluster registry specs.

Hardware accessors and Alive-node gating live here; policy math stays in gpu_alloc.
"""
from ray_hive.core import ray_utils
from ray_hive.core.gpu_alloc import (
    ConserveTdpAllocator,
    Fp8Allocator,
    PerformanceAllocator,
    TensorParallelAllocator,
)
from ray_hive.core.ray_utils import filter_alive


class _RayHardwareMixin:
    """Shared Ray registry hardware accessors + Alive-node gating."""

    def sm_count(self, gpu: dict) -> int:
        return ray_utils.sm_count(gpu)


    def compute_cap(self, gpu: dict) -> tuple[int, int]:
        return ray_utils.compute_cap(gpu)


    def approx_tdp(self, gpu: dict) -> float:
        return ray_utils.approx_tdp(gpu)


    def mem_bandwidth(self, gpu: dict) -> float:
        return ray_utils.mem_bandwidth(gpu)


    def filter_eligible(
        self,
        gpu_map: dict[str, dict],
        min_vram_gb: float,
        hf_params: dict,
        vllm_kwargs: dict,
    ) -> list[tuple[str, dict]]:
        eligible = super().filter_eligible(gpu_map, min_vram_gb, hf_params, vllm_kwargs)
        return filter_alive(eligible)


class RayPerformanceAllocator(_RayHardwareMixin, PerformanceAllocator):
    """Performance policy with Ray registry hardware + Alive-node gate."""


class RayFp8Allocator(_RayHardwareMixin, Fp8Allocator):
    """FP8 policy with Ray registry hardware + Alive-node gate."""


class RayConserveTdpAllocator(_RayHardwareMixin, ConserveTdpAllocator):
    """Conserve-TDP policy with Ray registry hardware + Alive-node gate."""


class RayTensorParallelAllocator(_RayHardwareMixin, TensorParallelAllocator):
    """Same-node TP packing with Ray registry hardware + Alive-node gate."""
