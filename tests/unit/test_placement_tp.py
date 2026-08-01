"""C — select_gpus / TP pin modes / shardability."""
import pytest

from ray_hive.core.ray_utils.placement import chunk_gpu_groups
from ray_hive.core.ray_utils.select_gpus import resolve_target_gpus
from ray_hive.core.ray_utils.tensor_parallel import assert_tp_shardable, tp_shardable
from ray_hive.errors import (
    GpuNotFoundError,
    InsufficientVramError,
    InvalidGpuPinError,
    PlacementError,
)
from tests.helpers import FakePerformanceAllocator, make_gpu, TINY_HF_DENSE


@pytest.fixture
def gmap():
    return {
        "host-a:gpu0": make_gpu("host-a:gpu0", 20, 24, sm=80, cap=(8, 9)),
        "host-a:gpu1": make_gpu("host-a:gpu1", 20, 24, sm=80, cap=(8, 9)),
        "host-b:gpu0": make_gpu("host-b:gpu0", 20, 24, sm=70, cap=(8, 9)),
    }


def test_pin_string(gmap, tiny_hf_dense):
    tp, gpus, _ = resolve_target_gpus(
        gmap, 1, "host-a:gpu0", tiny_hf_dense, FakePerformanceAllocator, None, {},
    )
    assert tp == 1 and len(gpus) == 1 and gpus[0]["gpu_key"] == "host-a:gpu0"


def test_pin_list_one(gmap, tiny_hf_dense):
    tp, gpus, _ = resolve_target_gpus(
        gmap, 1, ["host-b:gpu0"], tiny_hf_dense, FakePerformanceAllocator, None, {},
    )
    assert tp == 1 and gpus[0]["gpu_key"] == "host-b:gpu0"


def test_pin_tp_group_replicas_1(gmap, tiny_hf_dense):
    tp, gpus, _ = resolve_target_gpus(
        gmap, 1, ["host-a:gpu0", "host-a:gpu1"], tiny_hf_dense,
        FakePerformanceAllocator, None, {},
    )
    assert tp == 2 and len(gpus) == 2


def test_pin_multi_tp1(gmap, tiny_hf_dense):
    keys = ["host-a:gpu0", "host-b:gpu0"]
    tp, gpus, _ = resolve_target_gpus(
        gmap, 2, keys, tiny_hf_dense, FakePerformanceAllocator, None, {},
    )
    assert tp == 1 and len(gpus) == 2


def test_bad_replicas_for_list(gmap, tiny_hf_dense):
    with pytest.raises(InvalidGpuPinError):
        resolve_target_gpus(
            gmap, 3, ["host-a:gpu0", "host-a:gpu1"], tiny_hf_dense,
            FakePerformanceAllocator, None, {},
        )


def test_empty_gpu_list(gmap, tiny_hf_dense):
    with pytest.raises(InvalidGpuPinError):
        resolve_target_gpus(gmap, 1, [], tiny_hf_dense, FakePerformanceAllocator, None, {})


def test_tp_cross_host_rejected(gmap, tiny_hf_dense):
    with pytest.raises(PlacementError):
        resolve_target_gpus(
            gmap, 1, ["host-a:gpu0", "host-b:gpu0"], tiny_hf_dense,
            FakePerformanceAllocator, None, {},
        )


def test_missing_pin(gmap, tiny_hf_dense):
    with pytest.raises(GpuNotFoundError):
        resolve_target_gpus(
            gmap, 1, "missing:gpu0", tiny_hf_dense, FakePerformanceAllocator, None, {},
        )


def test_insufficient_pin(tiny_hf_dense):
    gmap = {"h:gpu0": make_gpu("h:gpu0", 0.01, 24)}
    with pytest.raises(InsufficientVramError):
        resolve_target_gpus(
            gmap, 1, "h:gpu0", tiny_hf_dense, FakePerformanceAllocator, None, {},
        )


def test_chunk_gpu_groups():
    gpus = [{"gpu_key": f"h:gpu{i}"} for i in range(4)]
    assert len(chunk_gpu_groups(gpus, 1)) == 4
    groups = chunk_gpu_groups(gpus, 2)
    assert len(groups) == 2
    with pytest.raises(PlacementError):
        chunk_gpu_groups(gpus[:3], 2)


def test_tp_shardable_heads():
    hf = {**TINY_HF_DENSE, "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 256}
    assert tp_shardable(hf, 2)
    assert_tp_shardable(hf, 2)
    bad = {**hf, "num_attention_heads": 5}
    assert not tp_shardable(bad, 2)
