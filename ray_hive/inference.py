"""
Standalone inference functions — route through ModelRouter deployments.

Non-stream calls use Serve DeploymentHandles. Streaming uses HTTP SSE to the
Serve OpenAI /v1/chat/completions endpoint (Ray Client cannot stream Serve handles).

Sync helpers (.result) are for non-async callers only. Inside an asyncio loop
use a_inference / a_inference_batch (Ray Serve 2.56+).
"""
import asyncio
import json
import os
from typing import Iterator, Optional, Type, List, Union

import ray
import requests
from pydantic import BaseModel
from ray import serve


def _ensure_connected():
    """Require an existing Ray connection (e.g. via RayHive(address=...))."""
    if not ray.is_initialized():
        raise RuntimeError("Ray is not connected. Call RayHive(address=...) first.")


def _get_handle(model_id: str):
    """Resolve ModelRouter deployment handle for a deployed model."""
    _ensure_connected()
    status = serve.status()
    if model_id not in status.applications:
        raise RuntimeError(f"Model '{model_id}' not found")
    app = status.applications[model_id]
    deployments = app.deployments if hasattr(app, 'deployments') else {}
    deployment_name = list(deployments.keys())[0]
    return serve.get_deployment_handle(deployment_name, app_name=model_id)


def _serve_base_url() -> str:
    """HTTP base for Serve (RAY_SERVE_URL or host from RAY_ADDRESS)."""
    explicit = os.environ.get("RAY_SERVE_URL")
    if explicit:
        return explicit.rstrip("/")
    addr = os.environ.get("RAY_ADDRESS", "")
    if addr.startswith("ray://"):
        return f"http://{addr.removeprefix('ray://').split(':')[0]}:8000"
    if ray.is_initialized():
        return "http://127.0.0.1:8000"
    raise RuntimeError("Set RAY_SERVE_URL or RAY_ADDRESS=ray://host:port for streaming")


def _assert_not_in_asyncio_loop():
    """Block sync DeploymentHandle APIs when already inside an asyncio loop."""
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False
    assert not in_loop, (
        "DeploymentHandle.result() cannot run inside an asyncio loop "
        "(Ray Serve 2.56+). Use a_inference / a_inference_batch instead."
    )


def _sync_result(response):
    """Block on a DeploymentResponse from sync code only (not inside asyncio)."""
    _assert_not_in_asyncio_loop()
    return response.result()


def _extract_text(result):
    """Extract text from vLLM result."""
    if isinstance(result, list):
        return result[0] if result else ""
    return str(result)


def _parse_structured_output(text: str, pydantic_class: Type[BaseModel]):
    """Parse JSON text into a Pydantic model instance."""
    return pydantic_class(**json.loads(text.strip()))


def _build_request(prompt=None, prompts=None, structured_output=None, max_tokens=None, **kwargs):
    """Build router infer() request dict."""
    request = {}
    if prompts is not None:
        request["prompts"] = prompts
    else:
        request["prompt"] = prompt
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    if structured_output:
        request["guided_json"] = structured_output.model_json_schema()
    request.update(kwargs)
    return request


def inference(prompt: str, model_id: str, structured_output: Optional[Type[BaseModel]] = None, max_tokens: Optional[int] = None, **kwargs) -> Union[str, BaseModel]:
    """Run inference on a deployed model. Ray Serve handles load balancing automatically."""
    handle = _get_handle(model_id)
    request = _build_request(prompt=prompt, structured_output=structured_output, max_tokens=max_tokens, **kwargs)
    result = _sync_result(handle.infer.remote(request))
    text = _extract_text(result)
    if structured_output:
        return _parse_structured_output(text, structured_output)
    return text


async def a_inference(prompt: str, model_id: str, structured_output: Optional[Type[BaseModel]] = None, max_tokens: Optional[int] = None, **kwargs) -> Union[str, BaseModel]:
    """Run async inference on a deployed model. Ray Serve handles load balancing automatically."""
    handle = _get_handle(model_id)
    request = _build_request(prompt=prompt, structured_output=structured_output, max_tokens=max_tokens, **kwargs)
    result = await handle.infer.remote(request)
    text = _extract_text(result)
    if structured_output:
        return _parse_structured_output(text, structured_output)
    return text


def inference_batch(prompts: List[str], model_id: str, structured_output: Optional[Type[BaseModel]] = None, max_tokens: Optional[int] = None, **kwargs) -> List[Union[str, BaseModel]]:
    """Run batch inference — one client request; router shards by measured replica tokens/sec."""
    handle = _get_handle(model_id)
    request = _build_request(prompts=prompts, structured_output=structured_output, max_tokens=max_tokens, **kwargs)
    result = _sync_result(handle.infer.remote(request))
    results = result if isinstance(result, list) else [result]
    return [
        _parse_structured_output(_extract_text(item), structured_output) if structured_output else _extract_text(item)
        for item in results
    ]


async def a_inference_batch(prompts: List[str], model_id: str, structured_output: Optional[Type[BaseModel]] = None, max_tokens: Optional[int] = None, **kwargs) -> List[Union[str, BaseModel]]:
    """Async batch inference — one client request; router shards by measured replica tokens/sec."""
    handle = _get_handle(model_id)
    request = _build_request(prompts=prompts, structured_output=structured_output, max_tokens=max_tokens, **kwargs)
    result = await handle.infer.remote(request)
    results = result if isinstance(result, list) else [result]
    return [
        _parse_structured_output(_extract_text(item), structured_output) if structured_output else _extract_text(item)
        for item in results
    ]


def inference_stream(prompt: str, model_id: str, max_tokens: Optional[int] = None, **kwargs) -> Iterator[str]:
    """Stream assistant deltas via HTTP SSE to /{model_id}/v1/chat/completions."""
    if kwargs.get("structured_output") is not None:
        raise ValueError("structured_output is not supported with inference_stream")
    _ensure_connected()
    if model_id not in serve.status().applications:
        raise RuntimeError(f"Model '{model_id}' not found")

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        **kwargs,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    url = f"{_serve_base_url()}/{model_id}/v1/chat/completions"
    with requests.post(url, json=body, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            data = raw[6:]
            if data == "[DONE]":
                break
            piece = (json.loads(data)["choices"][0].get("delta") or {}).get("content") or ""
            if piece:
                yield piece
