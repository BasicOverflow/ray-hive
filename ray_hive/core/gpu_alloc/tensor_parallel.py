"""
Tensor-parallel GPU allocator — not implemented yet.

TODO — vLLM only knows how to shard models on symmetric GPUs, not heterogeneous
ones, so come up with an algorithm to find an ideal set of GPUs that would spread
out a big model of size X such that the smallest GPU wouldn't limit the total
space deployable, even if that means fewer GPUs total. Because shards on each
GPU can only be as big as the smallest GPU's capacity
(usable ≈ N * min(available_i)). Prefer same-node sets (cross-node TP is painful
on this cluster). Return one multi-GPU placement, not N independent single-GPU
replicas. Not wired into deploy yet.
"""
from .base import BaseGpuAllocator


class TensorParallelAllocator(BaseGpuAllocator):
    """Placeholder for future symmetric TP set selection."""

    def score(
        self,
        gpu_key: str,
        gpu: dict,
        hf_params: dict,
        vllm_kwargs: dict,
    ) -> float:
        raise NotImplementedError("TP allocation not wired yet")


    def select(
        self,
        gpu_map: dict[str, dict],
        replicas: int,
        min_vram_gb: float,
        hf_params: dict,
        vllm_kwargs: dict,
    ) -> list[tuple[str, dict]]:
        """
        TODO — vLLM only knows how to shard models on symmetric GPUs, not
        heterogeneous ones, so come up with an algorithm to find an ideal set of
        GPUs that would spread out a big model of size X such that the smallest
        GPU wouldn't limit the total space deployable, even if that means fewer
        GPUs total. Because shards on each GPU can only be as big as the smallest
        GPU's capacity (usable ≈ N * min(available_i)). Prefer same-node sets;
        return one multi-GPU placement, not N independent replicas.
        """
        raise NotImplementedError("TP allocation not wired yet")
