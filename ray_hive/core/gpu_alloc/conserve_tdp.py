"""Conserve-TDP GPU allocator — prefer lower-power cards for long uptimes."""
from .base import BaseGpuAllocator


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
