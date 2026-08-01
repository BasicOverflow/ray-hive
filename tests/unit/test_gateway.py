"""J — OpenAI gateway helpers (mocked Serve)."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from ray_hive.core import openai_gateway as gw


def test_proxy_url():
    with patch.object(gw, "_head_http_base", return_value="http://head:8000"):
        assert gw._proxy_url("qwen", "chat/completions") == \
            "http://head:8000/qwen/v1/chat/completions"


def test_live_model_ids_filters_gateway_and_needs_router():
    running = MagicMock()
    running.status = MagicMock()
    running.status.name = "RUNNING"
    running.deployments = {"m1-router": object()}

    gateway = MagicMock()
    gateway.status = MagicMock()
    gateway.status.name = "RUNNING"
    gateway.deployments = {"openai-gateway": object()}

    dead = MagicMock()
    dead.status = MagicMock()
    dead.status.name = "DEPLOYING"
    dead.deployments = {"m2-router": object()}

    with patch.object(gw, "serve") as serve:
        serve.status.return_value.applications = {
            "hive-openai": gateway,
            "m1": running,
            "m2": dead,
        }
        assert gw._live_model_ids() == ["m1"]


def test_model_not_found_raises_http():
    with pytest.raises(HTTPException) as ei:
        gw._model_not_found("missing")
    assert ei.value.status_code == 404
