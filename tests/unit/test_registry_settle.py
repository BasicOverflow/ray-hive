"""Registry settle retry after kill / teardown (0 free lag)."""
from ray_hive.hive import _registry_still_settling, _retry_if_registry_empty
from ray_hive.errors import GpuNotFoundError, InsufficientVramError


def test_settling_markers():
    assert _registry_still_settling(GpuNotFoundError("gpu x not in registry. Known: []"))
    assert _registry_still_settling(
        InsufficientVramError("GPU h:gpu0 has 0.00GB available, need 0.81GB")
    )
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

    # patch sleep so unit stays fast
    import ray_hive.hive as hive_mod
    from unittest.mock import patch

    with patch.object(hive_mod.time, "sleep"):
        assert _retry_if_registry_empty(fn) == "ok"
    assert n["i"] == 3
