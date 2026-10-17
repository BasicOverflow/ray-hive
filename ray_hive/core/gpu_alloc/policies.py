"""Concrete single-GPU allocation policies (performance vs conserve-TDP)."""
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


class ConserveTdpAllocator(BaseGpuAllocator):
    """Rank eligible GPUs toward lower approximate TDP; SM as tie-break."""

    def score(
        self,
        gpu_key: str,
        gpu: dict,
        hf_params: dict,
        vllm_kwargs: dict,
    ) -> float:
        """Prefer lower TDP; among equals prefer higher SM count."""
        tdp = max(self.approx_tdp(gpu), 1.0)
        return -tdp + 1e-3 * self.sm_count(gpu)
