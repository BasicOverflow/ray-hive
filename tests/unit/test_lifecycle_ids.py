"""Duplicate model_id + shutdown helpers (mocked)."""
from unittest.mock import MagicMock, patch

import pytest

from ray_hive.core.ray_utils.lifecycle import assert_model_id_free
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
