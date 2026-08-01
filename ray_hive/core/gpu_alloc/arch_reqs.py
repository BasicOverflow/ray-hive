"""Architecture requirements derived from model quant / dtype."""

# Native FP8 (W8A8 / KV FP8 / Humming) needs Ada Lovelace SM 8.9+.
FP8_MIN_COMPUTE_CAP = (8, 9)


def _blob_looks_fp8(obj) -> bool:
    """True when nested config strings mention FP8 / float-quantized."""
    if isinstance(obj, dict):
        return any(_blob_looks_fp8(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_blob_looks_fp8(v) for v in obj)
    s = str(obj).lower()
    return any(x in s for x in ("fp8", "float8", "float-quantized"))


def model_needs_fp8_hardware(hf_params: dict, vllm_kwargs: dict) -> bool:
    """True when weights / KV / activations need native FP8 (SM 8.9+)."""
    for key in ("dtype", "kv_cache_dtype", "quantization"):
        if _blob_looks_fp8(vllm_kwargs.get(key)):
            return True
    for key in ("torch_dtype", "dtype", "kv_cache_dtype", "quantization_config"):
        if _blob_looks_fp8(hf_params.get(key)):
            return True
    return False


def required_min_compute_cap(
    hf_params: dict,
    vllm_kwargs: dict,
) -> tuple[int, int] | None:
    """Minimum (major, minor) compute capability, or None if unrestricted."""
    if model_needs_fp8_hardware(hf_params, vllm_kwargs):
        return FP8_MIN_COMPUTE_CAP
    return None
