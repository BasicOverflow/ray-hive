"""
Standalone inference functions — route through ModelRouter deployments.

Non-stream calls use Serve DeploymentHandles. Streaming uses HTTP SSE to the
Serve OpenAI /v1/chat/completions endpoint (Ray Client cannot stream Serve handles).

Sync helpers (.result) are for non-async callers only. Inside an asyncio loop
use a_inference / a_inference_batch (Ray Serve 2.56+).
"""
import asyncio
import json
from typing import Iterator, List, Optional, Type, Union

import ray
import requests
from pydantic import BaseModel
from ray import serve

from ray_hive.core.ray_utils import serve_base_url
from ray_hive.errors import InferenceError, ModelNotFoundError, UnsupportedModeError

PromptLike = Union[str, list, dict]


def _ensure_connected():
    """Require an existing Ray connection (e.g. via RayHive(address=...))."""
    if not ray.is_initialized():
        raise InferenceError("Ray is not connected. Call RayHive(address=...) first.")


def _get_handle(model_id: str):
    """Resolve ModelRouter deployment handle for a deployed model."""
    _ensure_connected()
    status = serve.status()
    if model_id not in status.applications:
        raise ModelNotFoundError(f"Model '{model_id}' not found", model_id=model_id)
    app = status.applications[model_id]
    deployments = app.deployments if hasattr(app, 'deployments') else {}
    deployment_name = list(deployments.keys())[0]
    return serve.get_deployment_handle(deployment_name, app_name=model_id)


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
    """Extract text from vLLM / router result."""
    if isinstance(result, list):
        if result and isinstance(result[0], list) and result and not isinstance(result[0], str):
            # embedding vector(s)
            return result[0] if len(result) == 1 else result
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


def inference(
    prompt: PromptLike,
    model_id: str,
    structured_output: Optional[Type[BaseModel]] = None,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> Union[str, BaseModel, list]:
    """Run inference on a deployed model (text, multimodal, or embeddings)."""
    handle = _get_handle(model_id)
    request = _build_request(prompt=prompt, structured_output=structured_output, max_tokens=max_tokens, **kwargs)
    result = _sync_result(handle.infer.remote(request))
    if structured_output:
        text = _extract_text(result)
        return _parse_structured_output(text if isinstance(text, str) else str(text), structured_output)
    if isinstance(result, list) and result and isinstance(result[0], list) and not isinstance(result[0], str):
        return result[0] if len(result) == 1 else result
    return _extract_text(result)


async def a_inference(
    prompt: PromptLike,
    model_id: str,
    structured_output: Optional[Type[BaseModel]] = None,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> Union[str, BaseModel, list]:
    """Run async inference on a deployed model."""
    handle = _get_handle(model_id)
    request = _build_request(prompt=prompt, structured_output=structured_output, max_tokens=max_tokens, **kwargs)
    result = await handle.infer.remote(request)
    if structured_output:
        text = _extract_text(result)
        return _parse_structured_output(text if isinstance(text, str) else str(text), structured_output)
    if isinstance(result, list) and result and isinstance(result[0], list) and not isinstance(result[0], str):
        return result[0] if len(result) == 1 else result
    return _extract_text(result)


def inference_batch(
    prompts: List[PromptLike],
    model_id: str,
    structured_output: Optional[Type[BaseModel]] = None,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> List[Union[str, BaseModel, list]]:
    """Run batch inference — one client request; router shards by planned max_num_seqs."""
    handle = _get_handle(model_id)
    request = _build_request(prompts=prompts, structured_output=structured_output, max_tokens=max_tokens, **kwargs)
    result = _sync_result(handle.infer.remote(request))
    results = result if isinstance(result, list) else [result]
    out = []
    for item in results:
        if structured_output:
            text = item if isinstance(item, str) else (item[0] if isinstance(item, list) and item and isinstance(item[0], str) else str(item))
            if isinstance(item, list) and item and isinstance(item[0], (int, float)):
                raise UnsupportedModeError("structured_output not supported for embeddings")
            out.append(_parse_structured_output(text if isinstance(text, str) else _extract_text(item), structured_output))
        else:
            out.append(item if isinstance(item, list) and item and isinstance(item[0], (int, float)) else _extract_text(item))
    return out


async def a_inference_batch(
    prompts: List[PromptLike],
    model_id: str,
    structured_output: Optional[Type[BaseModel]] = None,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> List[Union[str, BaseModel, list]]:
    """Async batch inference — one client request; router shards by planned max_num_seqs."""
    handle = _get_handle(model_id)
    request = _build_request(prompts=prompts, structured_output=structured_output, max_tokens=max_tokens, **kwargs)
    result = await handle.infer.remote(request)
    results = result if isinstance(result, list) else [result]
    out = []
    for item in results:
        if structured_output:
            out.append(_parse_structured_output(_extract_text(item) if not isinstance(item, str) else item, structured_output))
        else:
            out.append(item if isinstance(item, list) and item and isinstance(item[0], (int, float)) else _extract_text(item))
    return out


def inference_stream(prompt: PromptLike, model_id: str, max_tokens: Optional[int] = None, **kwargs) -> Iterator[str]:
    """Stream assistant deltas via HTTP SSE to /{model_id}/v1/chat/completions."""
    if kwargs.get("structured_output") is not None:
        raise UnsupportedModeError("structured_output is not supported with inference_stream")
    _ensure_connected()
    if model_id not in serve.status().applications:
        raise ModelNotFoundError(f"Model '{model_id}' not found", model_id=model_id)

    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        messages = prompt
    else:
        raise InferenceError("inference_stream prompt must be str or OpenAI messages list")

    body = {
        "model": model_id,
        "messages": messages,
        "stream": True,
        **kwargs,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    url = f"{serve_base_url()}/{model_id}/v1/chat/completions"
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
