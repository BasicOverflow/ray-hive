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


def test_router_model_card_uses_router_payload():
    class _Resp:
        def read(self):
            return b'{"object":"list","data":[{"id":"m1","context_window":4096,"max_output_tokens":512}]}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with patch.object(gw, "_proxy_url", return_value="http://head:8000/m1/v1/models"), \
            patch.object(gw.urllib.request, "urlopen", return_value=_Resp()):
        card = gw._router_model_card("m1")
    assert card["context_window"] == 4096
    assert card["max_output_tokens"] == 512


def test_router_model_card_fallback_on_error():
    with patch.object(gw, "_proxy_url", return_value="http://head:8000/m1/v1/models"), \
            patch.object(gw.urllib.request, "urlopen", side_effect=gw.urllib.error.URLError("down")):
        assert gw._router_model_card("m1") == gw._fallback_model_card("m1")
