"""
Base GPU allocator — placement policy contract.

Policies filter eligible GPUs, score them, and select top-N (or all).
Hardware accessors default to registry PyCUDA helpers in ray_utils.hardware.
"""
from ray_hive.core.ray_utils.hardware import (
    approx_tdp as _approx_tdp,
    compute_cap as _compute_cap,
    mem_bandwidth as _mem_bandwidth,
    sm_count as _sm_count,
)

from .arch_reqs import required_min_compute_cap

# Packing fraction for util pool (must match gpu_budget_frac in placement).
# Leave ~5% outside the util pool for CUDA-graph capture + sampler scratch.
TP1_BUDGET_FRAC = 0.95


class BaseGpuAllocator:
    """GPU placement policy base (override score / select in subclasses)."""

    def sm_count(self, gpu: dict) -> int:
        return _sm_count(gpu)

    def compute_cap(self, gpu: dict) -> tuple[int, int]:
        return _compute_cap(gpu)

    def approx_tdp(self, gpu: dict) -> float:
        return _approx_tdp(gpu)

    def mem_bandwidth(self, gpu: dict) -> float:
        return _mem_bandwidth(gpu)

    def filter_eligible(
        self,
        gpu_map: dict[str, dict],
        min_vram_gb: float,
        hf_params: dict,
        vllm_kwargs: dict,
    ) -> list[tuple[str, dict]]:
        """Return GPUs that fit weights and satisfy model arch requirements."""
        # plan_deployment only packs into available * TP1_BUDGET_FRAC.
        # Util capacity (total × frac) must also cover weight_need.
        need = min_vram_gb / TP1_BUDGET_FRAC
        min_cap = required_min_compute_cap(hf_params, vllm_kwargs)
        eligible = []
        for gpu_key, gpu in gpu_map.items():
            if float(gpu["total"]) * TP1_BUDGET_FRAC < min_vram_gb:
                continue
            if gpu["available"] < need:
                continue
            if min_cap is not None and self.compute_cap(gpu) < min_cap:
                continue
            eligible.append((gpu_key, gpu))
        return eligible

    def score(
        self,
        gpu_key: str,
        gpu: dict,
        hf_params: dict,
        vllm_kwargs: dict,
    ) -> float:
        """Return preference score (higher = better). Override in subclasses."""
        raise NotImplementedError

    @staticmethod
    def _is_unshared(gpu: dict) -> bool:
        """True when the GPU has no hive pending/active reservations."""
        return not gpu.get("pending") and not gpu.get("active")

    def select(
        self,
        gpu_map: dict[str, dict],
        replicas: int,
        min_vram_gb: float,
        hf_params: dict,
        vllm_kwargs: dict,
    ) -> list[tuple[str, dict]]:
        """Filter, prefer unshared GPUs, then pack; rank each tier by score."""
        eligible = self.filter_eligible(gpu_map, min_vram_gb, hf_params, vllm_kwargs)

        def _rank(items: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
            # Freest available first, then policy score (SM/TDP/…).
            return sorted(
                items,
                key=lambda item: (
                    float(item[1].get("available") or 0),
                    self.score(item[0], item[1], hf_params, vllm_kwargs),
                ),
                reverse=True,
            )

        unshared = _rank([(k, g) for k, g in eligible if self._is_unshared(g)])
        shared = _rank([(k, g) for k, g in eligible if not self._is_unshared(g)])
        ranked = unshared + shared
        if replicas == -1:
            return ranked
        return ranked[:replicas]
