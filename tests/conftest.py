"""Root pytest fixtures for ray-hive tests."""
import io
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

# Unit hosts (esp. Windows) may have a broken vLLM wheel; stub so ModelRouter imports.
def _stub_vllm_if_needed():
    try:
        from vllm import SamplingParams  # noqa: F401
        from vllm.sampling_params import StructuredOutputsParams  # noqa: F401
        return
    except Exception:
        pass
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            del sys.modules[name]

    class SamplingParams:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class StructuredOutputsParams:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    vllm = ModuleType("vllm")
    sampling = ModuleType("vllm.sampling_params")
    sampling.StructuredOutputsParams = StructuredOutputsParams
    vllm.SamplingParams = SamplingParams
    vllm.sampling_params = sampling
    sys.modules["vllm"] = vllm
    sys.modules["vllm.sampling_params"] = sampling


_stub_vllm_if_needed()

from tests.helpers import (
    TINY_HF_DENSE,
    TINY_HF_FP8,
    TINY_HF_MM,
    FakeConserveTdpAllocator,
    FakePerformanceAllocator,
    make_gpu,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_hf_dense():
    return dict(TINY_HF_DENSE)


@pytest.fixture
def tiny_hf_fp8():
    return dict(TINY_HF_FP8)


@pytest.fixture
def tiny_hf_mm():
    return dict(TINY_HF_MM)


@pytest.fixture
def gpu_map_hetero():
    """Two hosts, mixed sizes; one shared GPU; one low-cap card."""
    return {
        "host-a:gpu0": make_gpu("host-a:gpu0", 22.0, 24.0, sm=100, tdp=350, cap=(8, 9), name="Ada"),
        "host-a:gpu1": make_gpu(
            "host-a:gpu1", 6.0, 8.0, sm=60, tdp=200, cap=(8, 6), name="Ampere",
            pending={"other": 1.0},
        ),
        "host-b:gpu0": make_gpu("host-b:gpu0", 22.0, 24.0, sm=90, tdp=250, cap=(8, 9), name="Ada2"),
        "host-b:gpu1": make_gpu("host-b:gpu1", 7.0, 8.0, sm=50, tdp=180, cap=(7, 5), name="Turing"),
    }


@pytest.fixture
def perf_alloc():
    return FakePerformanceAllocator()


@pytest.fixture
def tdp_alloc():
    return FakeConserveTdpAllocator()


@pytest.fixture
def tiny_png_bytes():
    """Minimal valid 1x1 PNG."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow required")
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def hf_config_path(tmp_path, tiny_hf_dense):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(tiny_hf_dense))
    return p
