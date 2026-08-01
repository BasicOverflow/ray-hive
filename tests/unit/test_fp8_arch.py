"""K — arch hard-fail / FP8 hardware requirements."""
from ray_hive.core.gpu_alloc.arch_reqs import (
    FP8_MIN_COMPUTE_CAP,
    model_needs_fp8_hardware,
    required_min_compute_cap,
)


def test_fp8_min_cap_constant():
    assert FP8_MIN_COMPUTE_CAP == (8, 9)


def test_nested_quantization_config():
    hf = {"quantization_config": {"quant_method": "FP8", "bits": 8}}
    assert model_needs_fp8_hardware(hf, {})
    assert required_min_compute_cap(hf, {}) == (8, 9)


def test_case_insensitive_float8():
    assert model_needs_fp8_hardware({"torch_dtype": "Float8_e4m3fn"}, {})


def test_no_fp8_unrestricted():
    assert required_min_compute_cap({"torch_dtype": "bfloat16"}, {}) is None
