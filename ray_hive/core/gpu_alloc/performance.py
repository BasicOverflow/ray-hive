"""Performance GPU allocator — prefer highest SM count (and bandwidth)."""
from .base import BaseGpuAllocator


class PerformanceAllocator(BaseGpuAllocator):
    """Rank eligible GPUs by compute proxy; select top-N for replicas."""

    def score(
        self,
        gpu_key: str,
        gpu: dict,
        hf_params: dict,
        vllm_kwargs: dict,
    ) -> float:
        """Prefer more SMs; lightly blend in memory bandwidth."""
        return float(self.sm_count(gpu)) + 1e-6 * self.mem_bandwidth(gpu)
