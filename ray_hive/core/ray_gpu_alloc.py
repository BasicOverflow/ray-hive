"""
Ray-wired GPU allocators — bind abstract policies to cluster registry specs.

Hardware accessors and Alive-node gating live here; policy math stays in gpu_alloc.
"""
from ray_hive import ray_utils
from ray_hive.core.gpu_alloc import ConserveTdpAllocator, Fp8Allocator, PerformanceAllocator


def _alive_only(eligible: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Drop GPUs whose registry host is not an Alive Ray node."""
    return [(k, g) for k, g in eligible if ray_utils.is_node_alive(k.split(":")[0])]


class RayPerformanceAllocator(PerformanceAllocator):
    """Performance policy with Ray registry hardware + Alive-node gate."""

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
        return _alive_only(eligible)


class RayFp8Allocator(Fp8Allocator):
    """FP8 policy with Ray registry hardware + Alive-node gate."""

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
        return _alive_only(eligible)


class RayConserveTdpAllocator(ConserveTdpAllocator):
    """Conserve-TDP policy with Ray registry hardware + Alive-node gate."""

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
        return _alive_only(eligible)
