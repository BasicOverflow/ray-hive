"""
Ray-wired GPU allocators — bind abstract policies to cluster registry specs.

Hardware accessors, Alive-node gating, and cpu_ram weight-spill eligibility live here;
policy scoring stays in gpu_alloc.
"""
from ray_hive.core import ray_utils
from ray_hive.core.gpu_alloc import (
    ConserveTdpAllocator,
    Fp8Allocator,
    PerformanceAllocator,
    TensorParallelAllocator,
)
from ray_hive.core.gpu_alloc import on_gpu_weight_need_gb
from ray_hive.core.ray_utils.hardware import count_by_host, filter_alive, host_memory_available_gb


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
        cpu_ram_cfg = float(vllm_kwargs.get("cpu_ram_per_instance", 0) or 0)
        tp_size = max(1, int(vllm_kwargs.get("tensor_parallel_size", 1) or 1))
        if tp_size > 1 or cpu_ram_cfg == 0:
            return filter_alive(
                super().filter_eligible(gpu_map, min_vram_gb, hf_params, vllm_kwargs)
            )

        # Weight fit with host spill (-1 / hard GiB). min_vram_gb is raw weight need. TP=1 only.
        eligible = dict(gpu_map)
        for _ in range(4):
            host_n = count_by_host(eligible)
            nxt = {}
            for k, g in eligible.items():
                h = k.split(":")[0]
                avail = float(g["available"])
                try:
                    need = on_gpu_weight_need_gb(
                        min_vram_gb, avail, cpu_ram_cfg, host_memory_available_gb(h), host_n[h]
                    )
                except ValueError:
                    continue
                if avail >= need:
                    nxt[k] = g
            if set(nxt) == set(eligible):
                break
            eligible = nxt
        return filter_alive(list(eligible.items()))


class RayPerformanceAllocator(_RayHardwareMixin, PerformanceAllocator):
    """Performance policy with Ray registry hardware + Alive-node gate."""


class RayFp8Allocator(_RayHardwareMixin, Fp8Allocator):
    """FP8 policy with Ray registry hardware + Alive-node gate."""


class RayConserveTdpAllocator(_RayHardwareMixin, ConserveTdpAllocator):
    """Conserve-TDP policy with Ray registry hardware + Alive-node gate."""


class RayTensorParallelAllocator(_RayHardwareMixin, TensorParallelAllocator):
    """Same-node TP packing with Ray registry hardware + Alive-node gate."""
