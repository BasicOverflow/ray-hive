"""
Standalone inference functions — route through ModelRouter deployments.

All four public functions resolve the router handle for model_id and send
prompt dicts; the router picks the least-loaded replica.
"""
import ray
from ray import serve
from typing import Optional, Type, List, Union
from pydantic import BaseModel
from os import getenv

from .ray_utils import load_env


def _ensure_connected():
    """Ensure Ray is connected to cluster."""
    if not ray.is_initialized():
        load_env()
        address = getenv("RAY_ADDRESS")
        if not address:
            raise RuntimeError("RAY_ADDRESS not set. Copy .env.example to .env and set your cluster address.")
        ray.init(address=address, ignore_reinit_error=True, log_to_driver=False)


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


def _extract_text(result):
    """Extract text from vLLM result."""
    if isinstance(result, list):
        return result[0] if result else ""
    return str(result)


def _parse_structured_output(text: str, pydantic_class: Type[BaseModel]):
    """Parse JSON text into a Pydantic model instance."""
    import json
    return pydantic_class(**json.loads(text.strip()))


def inference(prompt: str, model_id: str, structured_output: Optional[Type[BaseModel]] = None, max_tokens: Optional[int] = None, **kwargs) -> Union[str, BaseModel]:
    """Run inference on a deployed model. Ray Serve handles load balancing automatically."""
    handle = _get_handle(model_id)
    
    request = {"prompt": prompt}
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    
    # Use vLLM's native guided_json for structured output
    if structured_output:
        request["guided_json"] = structured_output.model_json_schema()
    
    request.update(kwargs)
    
    result = handle.infer.remote(request).result()
    text = _extract_text(result)
    
    if structured_output:
        return _parse_structured_output(text, structured_output)
    
    return text


async def a_inference(prompt: str, model_id: str, structured_output: Optional[Type[BaseModel]] = None, max_tokens: Optional[int] = None, **kwargs) -> Union[str, BaseModel]:
    """Run async inference on a deployed model. Ray Serve handles load balancing automatically."""
    handle = _get_handle(model_id)
    
    request = {"prompt": prompt}
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    
    # Use vLLM's native guided_json for structured output
    if structured_output:
        request["guided_json"] = structured_output.model_json_schema()
    
    request.update(kwargs)
    
    result = await handle.infer.remote(request)
    text = _extract_text(result)
    
    if structured_output:
        return _parse_structured_output(text, structured_output)
    
    return text


def inference_batch(prompts: List[str], model_id: str, structured_output: Optional[Type[BaseModel]] = None, max_tokens: Optional[int] = None, **kwargs) -> List[Union[str, BaseModel]]:
    """Run batch inference on a deployed model. vLLM handles batching internally.
    
    All prompts are sent in a single request. vLLM's internal batching mechanism
    handles optimal batching based on max_num_seqs and max_num_batched_tokens.
    
    Args:
        prompts: List of prompts to process
        model_id: Model identifier
        structured_output: Optional Pydantic model for structured output
        max_tokens: Maximum tokens to generate
        **kwargs: Additional sampling parameters
    """
    handle = _get_handle(model_id)
    
    request = {"prompts": prompts}
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    
    # Use vLLM's native guided_json for structured output
    if structured_output:
        request["guided_json"] = structured_output.model_json_schema()
    
    request.update(kwargs)
    
    result = handle.infer.remote(request).result()
    results = result if isinstance(result, list) else [result]
    
    output = []
    for result_item in results:
        text = _extract_text(result_item)
        output.append(_parse_structured_output(text, structured_output) if structured_output else text)
    return output


async def a_inference_batch(prompts: List[str], model_id: str, structured_output: Optional[Type[BaseModel]] = None, max_tokens: Optional[int] = None, **kwargs) -> List[Union[str, BaseModel]]:
    """Run async batch inference on a deployed model. vLLM handles batching internally.
    
    All prompts are sent in a single request. vLLM's internal batching mechanism
    handles optimal batching based on max_num_seqs and max_num_batched_tokens.
    
    Args:
        prompts: List of prompts to process
        model_id: Model identifier
        structured_output: Optional Pydantic model for structured output
        max_tokens: Maximum tokens to generate
        **kwargs: Additional sampling parameters
    """
    handle = _get_handle(model_id)
    
    request = {"prompts": prompts}
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    
    # Use vLLM's native guided_json for structured output
    if structured_output:
        request["guided_json"] = structured_output.model_json_schema()
    
    request.update(kwargs)
    
    result = await handle.infer.remote(request)
    results = result if isinstance(result, list) else [result]
    
    output = []
    for result_item in results:
        text = _extract_text(result_item)
        output.append(_parse_structured_output(text, structured_output) if structured_output else text)
    return output

