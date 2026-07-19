"""
Base GPU allocator — abstract placement policy contract.

Policies filter eligible GPUs, score them, and select top-N (or all).
Hardware accessors (SM count, TDP, etc.) are abstract; Ray wiring fills them in.
"""
from abc import ABC, abstractmethod


class BaseGpuAllocator(ABC):
    """Abstract GPU placement policy."""

    @abstractmethod
    def sm_count(self, gpu: dict) -> int:
        """Return streaming multiprocessor count for a GPU view."""


    @abstractmethod
    def compute_cap(self, gpu: dict) -> tuple[int, int]:
        """Return (major, minor) compute capability."""


    @abstractmethod
    def approx_tdp(self, gpu: dict) -> float:
        """Return approximate TDP in watts for power-aware ranking."""


    @abstractmethod
    def mem_bandwidth(self, gpu: dict) -> float:
        """Return approximate memory bandwidth proxy (higher = better)."""


    def filter_eligible(
        self,
        gpu_map: dict[str, dict],
        min_vram_gb: float,
        hf_params: dict,
        vllm_kwargs: dict,
    ) -> list[tuple[str, dict]]:
        """Return GPUs with enough available VRAM for model weights."""
        eligible = []
        for gpu_key, gpu in gpu_map.items():
            if gpu["available"] < min_vram_gb:
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


    def select(
        self,
        gpu_map: dict[str, dict],
        replicas: int,
        min_vram_gb: float,
        hf_params: dict,
        vllm_kwargs: dict,
    ) -> list[tuple[str, dict]]:
        """Filter, rank by score descending, take top replicas (-1 = all)."""
        eligible = self.filter_eligible(gpu_map, min_vram_gb, hf_params, vllm_kwargs)
        ranked = sorted(
            eligible,
            key=lambda item: self.score(item[0], item[1], hf_params, vllm_kwargs),
            reverse=True,
        )
        if replicas == -1:
            return ranked
        return ranked[:replicas]
