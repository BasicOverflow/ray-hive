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


def test_mark_sleeping_holds_vram_after_free_bump(alloc):
    reserved = 8.0
    assert alloc.reserve_replica("r1", "host-a:gpu0", reserved)
    alloc.mark_initialized("r1", "host-a:gpu0")
    # Simulate level-1 sleep: nvidia-smi free rises toward total
    alloc.update_gpu("host-a", 0, free_gb=24.0, total_gb=24.0)
    assert alloc.get_gpu_vram("host-a:gpu0")["available"] == pytest.approx(24.0)

    assert alloc.mark_sleeping(["r1"]) == 1
    view = alloc.get_gpu_vram("host-a:gpu0")
    assert "r1" in view["pending"] and "r1" not in view["active"]
    assert view["available"] == pytest.approx(24.0 - reserved)
    assert not alloc.reserve_replica("r2", "host-a:gpu0", 20.0)


def test_mark_awake_releases_sleep_hold(alloc):
    assert alloc.reserve_replica("r1", "host-a:gpu0", 8.0)
    alloc.mark_initialized("r1", "host-a:gpu0")
    alloc.update_gpu("host-a", 0, free_gb=24.0, total_gb=24.0)
    alloc.mark_sleeping(["r1"])

    assert alloc.mark_awake(["r1"]) == 1
    view = alloc.get_gpu_vram("host-a:gpu0")
    assert "r1" in view["active"] and "r1" not in view["pending"]
    assert view["available"] == pytest.approx(24.0)


def test_mark_sleeping_idempotent_and_unknown(alloc):
    assert alloc.mark_sleeping(["missing"]) == 0
    assert alloc.reserve_replica("r1", "host-a:gpu0", 4.0)
    # still pending (not initialized) — nothing in active to move
    assert alloc.mark_sleeping(["r1"]) == 0
    alloc.mark_initialized("r1", "host-a:gpu0")
    assert alloc.mark_sleeping(["r1"]) == 1
    assert alloc.mark_sleeping(["r1"]) == 0


def test_mark_sleeping_tp_replica_on_two_gpus(alloc):
    alloc.update_gpu("host-a", 1, free_gb=10.0, total_gb=24.0)
    assert alloc.reserve_replica("r-tp", "host-a:gpu0", 6.0)
    assert alloc.reserve_replica("r-tp", "host-a:gpu1", 6.0)
    alloc.mark_initialized("r-tp", "host-a:gpu0")
    alloc.mark_initialized("r-tp", "host-a:gpu1")

    assert alloc.mark_sleeping(["r-tp"]) == 2
    for key in ("host-a:gpu0", "host-a:gpu1"):
        view = alloc.get_gpu_vram(key)
        assert "r-tp" in view["pending"] and "r-tp" not in view["active"]

    assert alloc.mark_awake(["r-tp"]) == 2
    for key in ("host-a:gpu0", "host-a:gpu1"):
        view = alloc.get_gpu_vram(key)
        assert "r-tp" in view["active"] and "r-tp" not in view["pending"]
