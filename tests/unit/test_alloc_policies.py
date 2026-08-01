"""A — allocation policies (no model deploy)."""
from unittest.mock import patch

import pytest

from ray_hive.core.gpu_alloc import TP1_BUDGET_FRAC, model_needs_fp8_hardware, required_min_compute_cap
from ray_hive.core.gpu_alloc.tensor_parallel import TensorParallelAllocator
from ray_hive.core.ray_gpu_alloc import RayPerformanceAllocator
from tests.helpers import (
    FakeConserveTdpAllocator,
    FakePerformanceAllocator,
    FakeAllocator,
    make_gpu,
)


def test_replicas_one_and_all(perf_alloc, tiny_hf_dense):
    gmap = {
        "h:gpu0": make_gpu("h:gpu0", 20, 24, sm=100),
        "h:gpu1": make_gpu("h:gpu1", 20, 24, sm=50),
    }
    one = perf_alloc.select(gmap, 1, 0.5, tiny_hf_dense, {})
    assert len(one) == 1 and one[0][0] == "h:gpu0"
    all_ = perf_alloc.select(gmap, -1, 0.5, tiny_hf_dense, {})
    assert len(all_) == 2


def test_partial_vram_eligibility(perf_alloc, tiny_hf_dense):
    # need = min_vram / 0.9; available must clear that and total*0.9 >= min_vram
    min_vram = 5.0
    need = min_vram / TP1_BUDGET_FRAC
    gmap = {
        "ok:gpu0": make_gpu("ok:gpu0", need + 0.1, 24, sm=80),
        "low:gpu0": make_gpu("low:gpu0", need - 0.5, 24, sm=100),
    }
    chosen = perf_alloc.select(gmap, -1, min_vram, tiny_hf_dense, {})
    assert [k for k, _ in chosen] == ["ok:gpu0"]


def test_prefer_unshared_then_shared(perf_alloc, tiny_hf_dense):
    gmap = {
        "shared:gpu0": make_gpu("shared:gpu0", 20, 24, sm=200, pending={"x": 1.0}),
        "free:gpu0": make_gpu("free:gpu0", 20, 24, sm=50),
    }
    chosen = perf_alloc.select(gmap, 1, 0.5, tiny_hf_dense, {})
    assert chosen[0][0] == "free:gpu0"
    both = perf_alloc.select(gmap, 2, 0.5, tiny_hf_dense, {})
    assert [k for k, _ in both] == ["free:gpu0", "shared:gpu0"]


def test_performance_sm_and_bandwidth_tiebreak(perf_alloc, tiny_hf_dense):
    gmap = {
        "a:gpu0": make_gpu("a:gpu0", 20, 24, sm=80, bandwidth=100),
        "b:gpu0": make_gpu("b:gpu0", 20, 24, sm=80, bandwidth=900),
    }
    assert perf_alloc.select(gmap, 1, 0.5, tiny_hf_dense, {})[0][0] == "b:gpu0"


def test_conserve_tdp_prefers_lower(tdp_alloc, tiny_hf_dense):
    gmap = {
        "hot:gpu0": make_gpu("hot:gpu0", 20, 24, sm=100, tdp=400),
        "cool:gpu0": make_gpu("cool:gpu0", 20, 24, sm=60, tdp=150),
    }
    assert tdp_alloc.select(gmap, 1, 0.5, tiny_hf_dense, {})[0][0] == "cool:gpu0"


def test_fp8_hardware_detection(tiny_hf_fp8, tiny_hf_dense):
    assert model_needs_fp8_hardware(tiny_hf_fp8, {})
    assert required_min_compute_cap(tiny_hf_fp8, {}) == (8, 9)
    assert not model_needs_fp8_hardware(tiny_hf_dense, {})
    assert model_needs_fp8_hardware({}, {"kv_cache_dtype": "fp8"})


def test_fp8_filters_low_compute_cap(perf_alloc, tiny_hf_fp8):
    gmap = {
        "ada:gpu0": make_gpu("ada:gpu0", 20, 24, sm=80, cap=(8, 9)),
        "old:gpu0": make_gpu("old:gpu0", 20, 24, sm=200, cap=(8, 6)),
    }
    chosen = perf_alloc.select(gmap, -1, 0.5, tiny_hf_fp8, {})
    assert [k for k, _ in chosen] == ["ada:gpu0"]


def test_no_eligible_returns_empty(perf_alloc, tiny_hf_dense):
    gmap = {"h:gpu0": make_gpu("h:gpu0", 0.1, 1.0, sm=80)}
    assert perf_alloc.select(gmap, 1, 5.0, tiny_hf_dense, {}) == []


class FakeTP(FakeAllocator, TensorParallelAllocator):
    pass


def test_tp_allocator_same_host_groups(tiny_hf_dense):
    alloc = FakeTP()
    gmap = {
        "a:gpu0": make_gpu("a:gpu0", 10, 12, sm=80),
        "a:gpu1": make_gpu("a:gpu1", 10, 12, sm=80),
        "b:gpu0": make_gpu("b:gpu0", 10, 12, sm=80),
    }
    out = alloc.select(gmap, 1, 0.5, tiny_hf_dense, {"tensor_parallel_size": 2})
    assert len(out) == 2
    hosts = {k.split(":")[0] for k, _ in out}
    assert hosts == {"a"}


def test_tp_requires_tp_size_ge_2(tiny_hf_dense):
    alloc = FakeTP()
    with pytest.raises(AssertionError):
        alloc.select({"a:gpu0": make_gpu("a:gpu0", 10, 12)}, 1, 0.5, tiny_hf_dense, {})


def test_ray_alive_filter(tiny_hf_dense):
    gmap = {
        "alive:gpu0": make_gpu("alive:gpu0", 20, 24, sm=80),
        "dead:gpu0": make_gpu("dead:gpu0", 20, 24, sm=200),
    }
    with patch("ray_hive.core.ray_utils.hardware.is_node_alive", side_effect=lambda h: h == "alive"):
        chosen = RayPerformanceAllocator().select(gmap, -1, 0.5, tiny_hf_dense, {})
    assert [k for k, _ in chosen] == ["alive:gpu0"]
