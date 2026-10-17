"""Registry settle retry after kill / teardown (0 free lag)."""
from unittest.mock import MagicMock, patch

import pytest

from ray_hive.errors import ConfigError, GpuNotFoundError, InsufficientVramError, PlacementError
from ray_hive.hive import RayHive, _registry_still_settling, _retry_if_registry_empty


def test_settling_markers():
    assert _registry_still_settling(GpuNotFoundError("gpu x not in registry. Known: []"))
    assert _registry_still_settling(
        InsufficientVramError("GPU h:gpu0 has 0.00GB available, need 0.81GB")
    )
    assert _registry_still_settling(PlacementError("No GPUs in registry — cannot place model"))
    assert not _registry_still_settling(
        InsufficientVramError("GPU h:gpu0 has 2.00GB available, need 8.00GB")
    )


def test_retry_polls_then_succeeds():
    n = {"i": 0}

    def fn():
        n["i"] += 1
        if n["i"] < 3:
            raise InsufficientVramError("GPU h:gpu0 has 0.00GB available, need 1GB")
        return "ok"

    import ray_hive.hive as hive_mod

    with patch.object(hive_mod.time, "sleep"):
        assert _retry_if_registry_empty(fn) == "ok"
    assert n["i"] == 3


def test_retry_on_empty_registry_placement():
    n = {"i": 0}

    def fn():
        n["i"] += 1
        if n["i"] < 2:
            raise PlacementError("No GPUs in registry — cannot place model")
        return "ok"

    import ray_hive.hive as hive_mod

    with patch.object(hive_mod.time, "sleep"):
        assert _retry_if_registry_empty(fn) == "ok"
    assert n["i"] == 2


def test_get_vram_state_wait_polls():
    hive = object.__new__(RayHive)
    states = [
        {},
        {"h:gpu0": {"free": 0.0, "available": 0.0, "total": 24.0}},
        {"h:gpu0": {"free": 20.0, "available": 20.0, "total": 24.0}},
    ]
    n = {"i": 0}

    def _get(*_a, **_k):
        i = min(n["i"], len(states) - 1)
        n["i"] += 1
        return states[i]

    with (
        patch("ray_hive.hive.get_gpu_registry", return_value=MagicMock()),
        patch("ray_hive.hive.ray.get", side_effect=_get),
        patch("ray_hive.hive.time.sleep"),
        patch("ray_hive.hive.time.time", side_effect=[0, 1, 2, 3, 4]),
    ):
        out = hive.get_vram_state(wait=True, timeout_s=40)
    assert "h:gpu0" in out
    assert out["h:gpu0"]["free"] == 20.0


def test_get_vram_state_wait_timeout():
    hive = object.__new__(RayHive)
    with (
        patch("ray_hive.hive.get_gpu_registry", return_value=MagicMock()),
        patch("ray_hive.hive.ray.get", return_value={}),
        patch("ray_hive.hive.time.sleep"),
        patch("ray_hive.hive.time.time", side_effect=[0, 1, 50]),
    ):
        with pytest.raises(ConfigError, match="VRAM registry empty"):
            hive.get_vram_state(wait=True, timeout_s=40)


def test_free_gpus_ranks_freest():
    hive = object.__new__(RayHive)
    state = {
        "busy:gpu0": {
            "free": 20.0, "available": 20.0, "total": 24.0,
            "pending": {}, "active": {"r1": 4.0},
        },
        "free:gpu0": {
            "free": 22.0, "available": 22.0, "total": 24.0,
            "pending": {}, "active": {},
        },
        "freer:gpu0": {
            "free": 24.0, "available": 24.0, "total": 24.0,
            "pending": {}, "active": {},
        },
    }
    with (
        patch.object(RayHive, "get_vram_state", return_value=state),
        patch("ray_hive.core.ray_utils.hardware.is_node_alive", return_value=True),
    ):
        keys = hive.free_gpus(min_free_gb=4.0)
    assert keys == ["freer:gpu0", "free:gpu0"]
