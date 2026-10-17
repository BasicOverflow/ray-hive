"""Duplicate model_id + hive-scoped shutdown helpers (mocked)."""
from unittest.mock import MagicMock, patch

import pytest

from ray_hive.core.gpu_registry import VRAMAllocator
from ray_hive.core.ray_utils.lifecycle import assert_model_id_free, shutdown_all, shutdown_model
from ray_hive.errors import ConfigError, ModelAlreadyDeployedError


def test_assert_model_id_free_serve():
    registry = MagicMock()
    with patch("ray.serve.status") as status, patch("ray.get", return_value=False):
        status.return_value.applications = {"m1": object()}
        with pytest.raises(ModelAlreadyDeployedError):
            assert_model_id_free("m1", registry)


def test_assert_model_id_free_registry():
    registry = MagicMock()
    with patch("ray.serve.status") as status, patch("ray.get", return_value=True):
        status.return_value.applications = {}
        with pytest.raises(ModelAlreadyDeployedError):
            assert_model_id_free("m1", registry)


def test_assert_model_id_free_ok():
    registry = MagicMock()
    with patch("ray.serve.status") as status, patch("ray.get", return_value=False):
        status.return_value.applications = {}
        assert_model_id_free("m1", registry)


def test_gateway_name_reserved():
    registry = MagicMock()
    with pytest.raises(ConfigError):
        assert_model_id_free("hive-openai", registry)


def test_shutdown_all_deletes_only_hive_apps():
    registry = MagicMock()
    deleted = []

    def _get(obj):
        name = str(obj)
        if "list_deployments" in name:
            return {
                "m1": {
                    "deployment_id": "m1",
                    "model_id": "m1",
                    "replicas": {"m1-host-gpu0": {"replica_id": "m1-host-gpu0"}},
                }
            }
        if "get_all_gpus" in name:
            return {"host:gpu0": {"pending": {}, "active": {"m1-host-gpu0": 4.0}}}
        if "clear_all" in name:
            return 1
        return None

    # Make remote() return distinguishable MagicMocks
    registry.list_deployments.remote.return_value = MagicMock(name="list_deployments")
    registry.get_all_gpus.remote.return_value = MagicMock(name="get_all_gpus")
    registry.clear_all.remote.return_value = MagicMock(name="clear_all")

    def _get2(obj):
        n = getattr(obj, "_mock_name", None) or str(obj)
        if n == "list_deployments":
            return {
                "m1": {
                    "deployment_id": "m1",
                    "model_id": "m1",
                    "replicas": {"m1-host-gpu0": {"replica_id": "m1-host-gpu0"}},
                }
            }
        if n == "get_all_gpus":
            return {"host:gpu0": {"pending": {}, "active": {"m1-host-gpu0": 4.0}}}
        if n == "clear_all":
            return 1
        return None

    status = MagicMock()
    status.applications = {
        "m1": object(),
        "m1-host-gpu0": object(),
        "hive-openai": object(),
        "markets": object(),
    }

    with (
        patch("ray_hive.core.gpu_registry.get_gpu_registry", return_value=registry),
        patch("ray.get", side_effect=_get2),
        patch("ray.serve.status", return_value=status),
        patch("ray.serve.delete", side_effect=lambda name: deleted.append(name)),
        patch("ray_hive.core.ray_utils.lifecycle.time.sleep"),
        patch("ray.serve.shutdown") as serve_shutdown,
    ):
        shutdown_all()

    assert "markets" not in deleted
    assert set(deleted) >= {"m1", "m1-host-gpu0", "hive-openai"}
    serve_shutdown.assert_not_called()


def test_reconcile_drops_ghosts():
    alloc = VRAMAllocator()
    alloc.update_gpu("host-a", 0, free_gb=20.0, total_gb=24.0)
    alloc.reserve_replica("dead-replica", "host-a:gpu0", 4.0)
    alloc.mark_initialized("dead-replica", "host-a:gpu0")
    alloc.reserve_deployment(
        "m1",
        {"dead-replica": {"host-a:gpu0": 4.0}},
        deployment_type="model",
        model_id="m1",
    )
    cleared = alloc.reconcile_with_serve([])
    assert cleared >= 1
    assert "dead-replica" not in alloc.get_gpu_vram("host-a:gpu0")["active"]
    assert not alloc.has_deployment("m1")


def test_reconcile_protects_replicas_when_router_live():
    alloc = VRAMAllocator()
    alloc.update_gpu("host-a", 0, free_gb=20.0, total_gb=24.0)
    alloc.reserve_replica("m1-host-gpu0", "host-a:gpu0", 8.0)
    alloc.mark_initialized("m1-host-gpu0", "host-a:gpu0")
    alloc.reserve_deployment(
        "m1",
        {"m1-host-gpu0": {"host-a:gpu0": 8.0}},
        deployment_type="model",
        model_id="m1",
    )
    # Router live but replica briefly missing from Serve status (sleep flicker).
    cleared = alloc.reconcile_with_serve(["m1", "hive-openai"])
    assert "m1-host-gpu0" in alloc.get_gpu_vram("host-a:gpu0")["active"]
    assert alloc.has_deployment("m1")
    assert cleared == 0


def test_shutdown_model_discovers_orphans_without_registry():
    registry = MagicMock()
    deleted = []
    registry.get_deployment.remote.return_value = MagicMock(name="get_deployment")
    registry.clear_replicas.remote.return_value = MagicMock(name="clear_replicas")
    registry.release_deployment.remote.return_value = MagicMock(name="release_deployment")

    def _get(obj):
        n = getattr(obj, "_mock_name", None) or str(obj)
        if n == "get_deployment":
            return None
        if n == "clear_replicas":
            return 1
        return None

    status = MagicMock()
    status.applications = {
        "m1": object(),
        "m1-ergos-02-nv-gpu0": object(),
        "markets": object(),
    }

    with (
        patch("ray.get", side_effect=_get),
        patch("ray.serve.status", return_value=status),
        patch("ray.serve.delete", side_effect=lambda name: deleted.append(name)),
        patch("ray_hive.core.ray_utils.lifecycle.time.sleep"),
        patch("ray_hive.core.gpu_registry.get_gpu_registry", return_value=registry),
    ):
        shutdown_model("m1")

    assert "markets" not in deleted
    assert "m1-ergos-02-nv-gpu0" in deleted
    assert "m1" in deleted
