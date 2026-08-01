"""O — placement filter replicas=-1 / pin edge cases (mocked autos)."""
import pytest

from ray_hive.core.ray_utils.select_gpus import resolve_target_gpus
from ray_hive.errors import PlacementError
from tests.helpers import FakePerformanceAllocator, make_gpu


def test_auto_place_replicas_shortfall(tiny_hf_dense):
    gmap = {
        "h:gpu0": make_gpu("h:gpu0", 20, 24, sm=80),
    }
    with pytest.raises(PlacementError):
        resolve_target_gpus(
            gmap, 3, None, tiny_hf_dense, FakePerformanceAllocator, None, {},
        )


def test_auto_place_empty_map(tiny_hf_dense):
    with pytest.raises(PlacementError):
        resolve_target_gpus(
            {}, 1, None, tiny_hf_dense, FakePerformanceAllocator, None, {},
        )


def test_auto_place_one(tiny_hf_dense):
    gmap = {"h:gpu0": make_gpu("h:gpu0", 20, 24, sm=80)}
    tp, gpus, _ = resolve_target_gpus(
        gmap, 1, None, tiny_hf_dense, FakePerformanceAllocator, None, {},
    )
    assert tp == 1 and len(gpus) == 1
