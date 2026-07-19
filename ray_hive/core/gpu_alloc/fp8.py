"""FP8 GPU allocator — prefer Ada / 40-series when the model uses FP8."""
from .performance import PerformanceAllocator


def _looks_fp8(value) -> bool:
    if value is None:
        return False
    return "fp8" in str(value).lower() or "float8" in str(value).lower()


def _model_wants_fp8(hf_params: dict, vllm_kwargs: dict) -> bool:
    """True when vLLM kwargs or HF config request FP8 weights / KV."""
    for key in ("dtype", "kv_cache_dtype", "quantization"):
        if _looks_fp8(vllm_kwargs.get(key)):
            return True
    for key in ("torch_dtype", "dtype", "kv_cache_dtype", "quantization_config"):
        val = hf_params.get(key)
        if isinstance(val, dict):
            if any(_looks_fp8(v) for v in val.values()):
                return True
        elif _looks_fp8(val):
            return True
    return False


class Fp8Allocator(PerformanceAllocator):
    """Same ranking as PerformanceAllocator; prefer Ada GPUs when FP8 is used."""

    def filter_eligible(
        self,
        gpu_map: dict[str, dict],
        min_vram_gb: float,
        hf_params: dict,
        vllm_kwargs: dict,
    ) -> list[tuple[str, dict]]:
        """Prefer compute_cap >= (8, 9) when FP8 is requested and such GPUs exist."""
        eligible = super().filter_eligible(gpu_map, min_vram_gb, hf_params, vllm_kwargs)
        if not _model_wants_fp8(hf_params, vllm_kwargs):
            return eligible
        ada = [(k, g) for k, g in eligible if self.compute_cap(g) >= (8, 9)]
        return ada if ada else eligible
