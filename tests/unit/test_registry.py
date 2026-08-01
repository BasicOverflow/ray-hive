"""Registry pending/active / double-book (in-process VRAMAllocator)."""
import pytest

from ray_hive.core.gpu_registry import VRAMAllocator
from ray_hive.errors import InsufficientVramError


@pytest.fixture
def alloc():
    a = VRAMAllocator()
    a.update_gpu("host-a", 0, free_gb=10.0, total_gb=24.0)
    return a


def test_pending_reduces_available(alloc):
    assert alloc.reserve_replica("r1", "host-a:gpu0", 4.0)
    view = alloc.get_gpu_vram("host-a:gpu0")
    assert view["available"] == pytest.approx(6.0)
    alloc.mark_initialized("r1", "host-a:gpu0")
    view = alloc.get_gpu_vram("host-a:gpu0")
    assert "r1" in view["active"] and "r1" not in view["pending"]


def test_double_book_fails(alloc):
    assert alloc.reserve_replica("r1", "host-a:gpu0", 8.0)
    assert not alloc.reserve_replica("r2", "host-a:gpu0", 8.0)


def test_reject_ray_hex_node_id(alloc):
    with pytest.raises(AssertionError):
        alloc.update_gpu("a" * 40, 0, 8.0, 24.0)


def test_reserve_deployment_capacity(alloc):
    alloc.reserve_replica("r1", "host-a:gpu0", 3.0)
    alloc.reserve_deployment(
        "m1",
        {"r1": {"host-a:gpu0": 3.0}},
        deployment_type="model",
        model_id="m1",
    )
    assert alloc.has_deployment("m1")
    with pytest.raises(InsufficientVramError):
        alloc.reserve_deployment(
            "m2",
            {"r2": {"host-a:gpu0": 20.0}},
            deployment_type="model",
            model_id="m2",
        )


def test_clear_replicas(alloc):
    alloc.reserve_replica("r1", "host-a:gpu0", 2.0)
    alloc.mark_initialized("r1", "host-a:gpu0")
    assert alloc.clear_replicas(["r1"]) == 1
    assert alloc.get_gpu_vram("host-a:gpu0")["available"] == pytest.approx(10.0)
