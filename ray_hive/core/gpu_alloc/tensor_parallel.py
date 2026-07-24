"""
Tensor-parallel GPU allocator — same-node TP group packing.

vLLM shards evenly across GPUs, so usable VRAM ≈ N * min(available_i).
Prefer same-node sets; return a flat list of (gpu_key, gpu) in contiguous
groups of tp_size (deploy chunks by tensor_parallel_size).
"""
from itertools import combinations

from .base import BaseGpuAllocator


class TensorParallelAllocator(BaseGpuAllocator):
    """Select same-node GPU sets for tensor-parallel replicas."""

    def select(
        self,
        gpu_map: dict[str, dict],
        replicas: int,
        min_vram_gb: float,
        hf_params: dict,
        vllm_kwargs: dict,
    ) -> list[tuple[str, dict]]:
        """
        Pack same-node GPU groups of size tensor_parallel_size.

        Returns a flat list of length replicas * tp_size (or all non-overlapping
        groups when replicas=-1), ordered as contiguous TP groups.
        """
        tp_size = int(vllm_kwargs.get("tensor_parallel_size", 1))
        assert tp_size >= 2, f"TensorParallelAllocator requires tensor_parallel_size >= 2, got {tp_size}"

        eligible = self.filter_eligible(gpu_map, min_vram_gb, hf_params, vllm_kwargs)
        by_host: dict[str, list[tuple[str, dict]]] = {}
        for gpu_key, gpu in eligible:
            by_host.setdefault(gpu_key.split(":")[0], []).append((gpu_key, gpu))

        candidates: list[tuple[tuple, list[tuple[str, dict]]]] = []
        for gpus in by_host.values():
            if len(gpus) < tp_size:
                continue
            for combo in combinations(gpus, tp_size):
                group = list(combo)
                avails = [g["available"] for _, g in group]
                min_avail = min(avails)
                leftover = sum(a - min_avail for a in avails)
                totals = [g["total"] for _, g in group]
                total_spread = max(totals) - min(totals)
                sm_sum = sum(self.sm_count(g) for _, g in group)
                # Prefer matched card sizes (vLLM TP wants symmetric GPUs), then capacity.
                rank = (-total_spread, min_avail, -leftover, sm_sum)
                candidates.append((rank, group))

        candidates.sort(key=lambda item: item[0], reverse=True)

        used: set[str] = set()
        result: list[tuple[str, dict]] = []
        for _, group in candidates:
            keys = {k for k, _ in group}
            if keys & used:
                continue
            result.extend(group)
            used |= keys
            if replicas != -1 and len(result) // tp_size >= replicas:
                break

        if replicas == -1:
            return result
        return result[: replicas * tp_size]
