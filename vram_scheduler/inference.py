"""Standalone inference functions for VRAM scheduler."""
import ray
from ray import serve
from typing import Optional, Type, List, Union, AsyncGenerator
from pydantic import BaseModel
from .utils.ray_utils import init_ray


def _ensure_connected():
    """Ensure Ray is connected to cluster."""
    if not ray.is_initialized():
        import os
        address = os.getenv("RAY_ADDRESS", "ray://10.0.1.53:10001")
        ray.init(address=address, ignore_reinit_error=True, log_to_driver=False)


def _get_handle(model_id: str):
    """Get a handle to the model application. Ray Serve handles load balancing automatically."""
    _ensure_connected()
    status = serve.status()
    
    # Try to find the model as a single application first
    if model_id in status.applications:
        app = status.applications[model_id]
        deployments = app.deployments if hasattr(app, 'deployments') else app.get('deployments', {})
        if deployments:
            # Get handle to first deployment - Ray Serve automatically load balances
            # across all deployments in the application
            deployment_name = list(deployments.keys())[0]
            return serve.get_deployment_handle(deployment_name, app_name=model_id)
    
    # Fallback: look for deployments in apps with model_id prefix (legacy support)
    # Format: {model_id}-{gpu_name}
    matching_deployments = []
    for app_name, app in status.applications.items():
        if app_name.startswith(f"{model_id}-"):
            deployments = app.deployments if hasattr(app, 'deployments') else app.get('deployments', {})
            for deployment_name in deployments.keys():
                if deployment_name.startswith(f"{model_id}-"):
                    matching_deployments.append((deployment_name, app_name))
    
    if not matching_deployments:
        available_apps = list(status.applications.keys())
        raise RuntimeError(f"Model '{model_id}' not found. Available: {available_apps}")
    
    # Get handle to first matching deployment - Ray Serve handles load balancing
    deployment_name, app_name = matching_deployments[0]
    return serve.get_deployment_handle(deployment_name, app_name=app_name)


def _get_max_num_seqs(model_id: str) -> int:
    """Query the model's max_num_seqs value for optimal batch sizing."""
    handle = _get_handle(model_id)
    try:
        max_num_seqs = handle.get_max_num_seqs.remote().result()
        if max_num_seqs is None or max_num_seqs <= 0:
            return 32  # Default fallback
        return max_num_seqs
    except Exception as e:
        # Fallback if query fails
        return 32


def _extract_text(result):
    """Extract text from vLLM result."""
    if isinstance(result, list):
        return result[0] if result else ""
    return str(result)


def _parse_structured_output(text: str, pydantic_class: Type[BaseModel]):
    """Parse structured output from text."""
    import json
    import re
    
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            return pydantic_class(**parsed)
        except:
            pass
    
    try:
        parsed = json.loads(text)
        return pydantic_class(**parsed)
    except:
        raise ValueError(f"Could not parse structured output from: {text}")


def inference(
    prompt: str,
    model_id: str,
    structured_output: Optional[Type[BaseModel]] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> Union[str, BaseModel]:
    """Run inference on a deployed model. Ray Serve handles load balancing automatically."""
    handle = _get_handle(model_id)
    
    request = {"prompt": prompt}
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    request.update(kwargs)
    
    result = handle.remote(request).result()
    text = _extract_text(result)
    
    if structured_output:
        return _parse_structured_output(text, structured_output)
    
    return text


async def a_inference(
    prompt: str,
    model_id: str,
    structured_output: Optional[Type[BaseModel]] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> Union[str, BaseModel]:
    """Run async inference on a deployed model. Ray Serve handles load balancing automatically."""
    handle = _get_handle(model_id)
    
    request = {"prompt": prompt}
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    request.update(kwargs)
    
    result = await handle.remote(request)
    text = _extract_text(result)
    
    if structured_output:
        return _parse_structured_output(text, structured_output)
    
    return text


def inference_batch(
    prompts: List[str],
    model_id: str,
    structured_output: Optional[Type[BaseModel]] = None,
    max_tokens: Optional[int] = None,
    batch_size: Optional[int] = None,
    **kwargs
) -> List[Union[str, BaseModel]]:
    """Run batch inference on a deployed model. Ray Serve handles load balancing automatically.
    
    Batch size is automatically calculated based on the model's max_num_seqs for optimal performance.
    Prompts are automatically split into optimal batches if they exceed max_num_seqs.
    
    Args:
        prompts: List of prompts to process
        model_id: Model identifier
        structured_output: Optional Pydantic model for structured output
        max_tokens: Maximum tokens to generate
        batch_size: Number of prompts to send per request (default: auto-calculated from max_num_seqs)
        **kwargs: Additional sampling parameters
    """
    handle = _get_handle(model_id)
    
    request_template = {}
    if max_tokens is not None:
        request_template["max_tokens"] = max_tokens
    request_template.update(kwargs)
    
    # Auto-calculate batch size based on model's max_num_seqs if not specified
    if batch_size is None:
        max_num_seqs = _get_max_num_seqs(model_id)
        batch_size = max_num_seqs
    
    # Group prompts into optimal batches (respecting max_num_seqs)
    batches = []
    for i in range(0, len(prompts), batch_size):
        batches.append(prompts[i:i + batch_size])
    
    # Send batches - Ray Serve handles load balancing
    requests = []
    for batch in batches:
        requests.append(handle.remote({"prompts": batch, **request_template}))
    
    # Collect results
    batch_results = [req.result() for req in requests]
    
    # Flatten results
    output = []
    for batch_result in batch_results:
        if isinstance(batch_result, list):
            for result in batch_result:
                text = _extract_text(result)
                if structured_output:
                    output.append(_parse_structured_output(text, structured_output))
                else:
                    output.append(text)
        else:
            text = _extract_text(batch_result)
            if structured_output:
                output.append(_parse_structured_output(text, structured_output))
            else:
                output.append(text)
    
    return output


async def a_inference_batch(
    prompts: List[str],
    model_id: str,
    structured_output: Optional[Type[BaseModel]] = None,
    max_tokens: Optional[int] = None,
    batch_size: Optional[int] = None,
    **kwargs
) -> List[Union[str, BaseModel]]:
    """Run async batch inference on a deployed model. Ray Serve handles load balancing automatically.
    
    Batch size is automatically calculated based on the model's max_num_seqs for optimal performance.
    Prompts are automatically split into optimal batches if they exceed max_num_seqs.
    
    Args:
        prompts: List of prompts to process
        model_id: Model identifier
        structured_output: Optional Pydantic model for structured output
        max_tokens: Maximum tokens to generate
        batch_size: Number of prompts to send per request (default: auto-calculated from max_num_seqs)
        **kwargs: Additional sampling parameters
    """
    handle = _get_handle(model_id)
    
    request_template = {}
    if max_tokens is not None:
        request_template["max_tokens"] = max_tokens
    request_template.update(kwargs)
    
    # Auto-calculate batch size based on model's max_num_seqs if not specified
    if batch_size is None:
        max_num_seqs = _get_max_num_seqs(model_id)
        batch_size = max_num_seqs
    
    # Group prompts into optimal batches (respecting max_num_seqs)
    batches = []
    for i in range(0, len(prompts), batch_size):
        batches.append(prompts[i:i + batch_size])
    
    # Send batches - Ray Serve handles load balancing
    requests = []
    for batch in batches:
        requests.append(handle.remote({"prompts": batch, **request_template}))
    
    # Collect results
    batch_results = [await req for req in requests]
    
    # Flatten results
    output = []
    for batch_result in batch_results:
        if isinstance(batch_result, list):
            for result in batch_result:
                text = _extract_text(result)
                if structured_output:
                    output.append(_parse_structured_output(text, structured_output))
                else:
                    output.append(text)
        else:
            text = _extract_text(batch_result)
            if structured_output:
                output.append(_parse_structured_output(text, structured_output))
            else:
                output.append(text)
    
    return output


async def streaming_batch(
    prompts: List[str],
    model_id: str,
    max_tokens: Optional[int] = None,
    **kwargs
) -> AsyncGenerator[List[str], None]:
    """Stream batch inference results (async generator)."""
    raise NotImplementedError("Streaming not yet implemented")
