"""Cluster-wide OpenAI-compatible gateway for OpenWebUI (and similar clients).

Mounted at Serve route ``/v1`` so one connection URL lists every live hive
model_id and proxies ``/chat/completions``, ``/completions``, ``/embeddings``
to ``/{model_id}/v1/...``.
"""
import json
import urllib.error
import urllib.request

import ray
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from ray import serve

from ray_hive.errors import ModelNotFoundError, http_status_for

GATEWAY_APP = "hive-openai"
GATEWAY_ROUTE = "/v1"

app = FastAPI(title="ray-hive OpenAI gateway")


def _model_not_found(model_id: str):
    err = ModelNotFoundError(f"model {model_id!r} not found", model_id=model_id)
    raise HTTPException(status_code=http_status_for(err), detail=str(err)) from err


def _head_http_base() -> str:
    """HTTP base for the HeadOnly Serve proxy."""
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        resources = node.get("Resources") or {}
        if resources.get("node:__internal_head__"):
            return f"http://{node['NodeManagerAddress']}:8000"
        if node.get("IsHeadNode"):
            return f"http://{node['NodeManagerAddress']}:8000"
    return "http://127.0.0.1:8000"


def _live_model_ids() -> list[str]:
    """Live hive model_ids = RUNNING Serve apps that own a ``{id}-router`` deployment."""
    apps = serve.status().applications or {}
    live = []
    for name, app in apps.items():
        if name == GATEWAY_APP:
            continue
        status = getattr(app.status, "name", None) or str(app.status)
        if "RUNNING" not in status.upper():
            continue
        deps = app.deployments or {}
        if f"{name}-router" in deps:
            live.append(name)
    return live


def _proxy_url(model_id: str, suffix: str) -> str:
    return f"{_head_http_base()}/{model_id}/v1/{suffix.lstrip('/')}"


@serve.deployment(
    name="openai-gateway",
    ray_actor_options={"num_cpus": 0.1},
    autoscaling_config=None,
    num_replicas=1,
    max_ongoing_requests=100,
)
@serve.ingress(app)
class OpenAIGateway:
    """OpenAI /v1 facade over all live hive ModelRouter apps."""

    @app.get("/models")
    async def list_models(self):
        """OpenAI model list — one entry per live hive model_id."""
        data = [
            {"id": mid, "object": "model", "owned_by": "ray-hive"}
            for mid in _live_model_ids()
        ]
        return {"object": "list", "data": data}


    @app.get("/models/{model_id}")
    async def get_model(self, model_id: str):
        """OpenAI single-model lookup."""
        if model_id not in _live_model_ids():
            _model_not_found(model_id)
        return {"id": model_id, "object": "model", "owned_by": "ray-hive"}


    @app.post("/chat/completions")
    async def chat_completions(self, request: Request):
        """Proxy chat to ``/{model}/v1/chat/completions``."""
        return await self._proxy(request, "chat/completions")


    @app.post("/completions")
    async def completions(self, request: Request):
        """Proxy completions to ``/{model}/v1/completions``."""
        return await self._proxy(request, "completions")


    @app.post("/embeddings")
    async def embeddings(self, request: Request):
        """Proxy embeddings to ``/{model}/v1/embeddings``."""
        return await self._proxy(request, "embeddings")


    async def _proxy(self, request: Request, suffix: str):
        body = await request.body()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"invalid JSON: {e}") from e
        model_id = payload.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="model is required")
        if model_id not in _live_model_ids():
            _model_not_found(model_id)

        url = _proxy_url(model_id, suffix)
        stream = bool(payload.get("stream"))
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": request.headers.get("accept", "*/*")},
        )
        try:
            upstream = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            detail = e.read()
            return Response(content=detail, status_code=e.code, media_type=e.headers.get("Content-Type", "application/json"))
        except urllib.error.URLError as e:
            raise HTTPException(status_code=502, detail=f"upstream unreachable: {e}") from e

        if stream:
            def gen():
                try:
                    while True:
                        chunk = upstream.read(1024)
                        if not chunk:
                            break
                        yield chunk
                finally:
                    upstream.close()

            return StreamingResponse(
                gen(),
                media_type=upstream.headers.get("Content-Type", "text/event-stream"),
                status_code=upstream.status,
            )

        try:
            content = upstream.read()
        finally:
            upstream.close()
        return Response(
            content=content,
            status_code=upstream.status,
            media_type=upstream.headers.get("Content-Type", "application/json"),
        )


def ensure_openai_gateway(force: bool = False) -> None:
    """Start the ``/v1`` gateway Serve app if it is not already running.

    Runs ``serve.run`` inside a cluster task so FastAPI/pydantic are imported on
    the worker (avoids client↔cluster cloudpickle schema mismatches).
    """
    apps = serve.status().applications or {}
    if GATEWAY_APP in apps:
        status = getattr(apps[GATEWAY_APP].status, "name", None) or str(apps[GATEWAY_APP].status)
        if "RUNNING" in status.upper() and not force:
            return
        serve.delete(GATEWAY_APP)

    @ray.remote(num_cpus=0.1)
    def _deploy_gateway():
        from ray import serve as _serve
        from ray_hive.core.openai_gateway import GATEWAY_APP as name
        from ray_hive.core.openai_gateway import GATEWAY_ROUTE as route
        from ray_hive.core.openai_gateway import OpenAIGateway

        _serve.run(OpenAIGateway.bind(), name=name, route_prefix=route)
        return True

    ray.get(_deploy_gateway.remote())
