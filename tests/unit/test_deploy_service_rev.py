from unittest.mock import MagicMock, patch

from ray_hive.core.deployment import HIVE_DEPLOY_CODE_REV, get_deploy_service


def test_get_deploy_service_kills_stale_rev():
    stale = MagicMock()
    fresh = MagicMock()
    with (
        patch("ray_hive.core.deployment.ray_namespace", return_value="ray_hive"),
        patch("ray_hive.core.deployment.ray.get_actor", return_value=stale),
        patch("ray_hive.core.deployment.ray.get", return_value=HIVE_DEPLOY_CODE_REV - 1),
        patch("ray_hive.core.deployment.ray.kill") as kill,
        patch("ray_hive.core.deployment.time.sleep"),
        patch("ray_hive.core.deployment.DeployService") as cls,
    ):
        cls.options.return_value.remote.return_value = fresh
        assert get_deploy_service() is fresh
        kill.assert_called_once_with(stale)


def test_get_deploy_service_keeps_current_rev():
    current = MagicMock()
    with (
        patch("ray_hive.core.deployment.ray_namespace", return_value="ray_hive"),
        patch("ray_hive.core.deployment.ray.get_actor", return_value=current),
        patch("ray_hive.core.deployment.ray.get", return_value=HIVE_DEPLOY_CODE_REV),
        patch("ray_hive.core.deployment.ray.kill") as kill,
    ):
        assert get_deploy_service() is current
        kill.assert_not_called()
