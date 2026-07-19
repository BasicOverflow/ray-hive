"""
Standalone inference functions — route through ModelRouter deployments.

All four public functions resolve the router handle for model_id and send
prompt dicts; the router picks the least-loaded replica.

Sync helpers (.result) are for non-async callers only. Inside an asyncio loop
use a_inference / a_inference_batch (Ray Serve 2.56+).
"""
import asyncio
import ray
from ray import serve
from typing import Optional, Type, List, Union
from pydantic import BaseModel


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


def _sync_result(response):
    """Block on a DeploymentResponse from sync code only (not inside asyncio)."""
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False
    assert not in_loop, (
        "DeploymentHandle.result() cannot run inside an asyncio loop "
        "(Ray Serve 2.56+). Use a_inference / a_inference_batch instead."
    )
    return response.result()


def _extract_text(result):
    """Extract text from vLLM result."""
    if isinstance(result, list):
        return result[0] if result else ""
    return str(result)


def _parse_structured_output(text: str, pydantic_class: Type[BaseModel]):
    """Parse JSON text into a Pydantic model instance."""
    import json
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
