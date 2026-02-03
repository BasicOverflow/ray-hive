import ray
from ray import serve
from typing import Optional, Type, List, Union
from pydantic import BaseModel


def _ensure_connected():
    """Ensure Ray is connected to cluster."""
    if not ray.is_initialized():
        import os
        address = os.getenv("RAY_ADDRESS", "ray://10.0.1.53:10001")
        ray.init(address=address, ignore_reinit_error=True, log_to_driver=False)


def _get_handle(model_id: str):
    _ensure_connected()
    status = serve.status()
    
    if model_id in status.applications:
        app = status.applications[model_id]
        deployments = app.deployments if hasattr(app, 'deployments') else {}
        if deployments:
            return serve.get_deployment_handle(list(deployments.keys())[0], app_name=model_id)
    
    for app_name, app in status.applications.items():
        if app_name.startswith(f"{model_id}-"):
            deployments = app.deployments if hasattr(app, 'deployments') else {}
            for deployment_name in deployments.keys():
                if deployment_name.startswith(f"{model_id}-"):
                    return serve.get_deployment_handle(deployment_name, app_name=app_name)
    
    raise RuntimeError(f"Model '{model_id}' not found")


def _extract_text(result):
    """Extract text from vLLM result."""
    if isinstance(result, list):
        return result[0] if result else ""
    return str(result)


def _parse_structured_output(text: str, pydantic_class: Type[BaseModel]):
    import json
    return pydantic_class(**json.loads(text.strip()))


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
    
    # Use vLLM's native guided_json for structured output
    if structured_output:
        request["guided_json"] = structured_output.model_json_schema()
    
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
    
    # Use vLLM's native guided_json for structured output
    if structured_output:
        request["guided_json"] = structured_output.model_json_schema()
    
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
    **kwargs
) -> List[Union[str, BaseModel]]:
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
    
    result = handle.remote(request).result()
    results = result if isinstance(result, list) else [result]
    
    output = []
    for result_item in results:
        text = _extract_text(result_item)
        output.append(_parse_structured_output(text, structured_output) if structured_output else text)
    return output


async def a_inference_batch(
    prompts: List[str],
    model_id: str,
    structured_output: Optional[Type[BaseModel]] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> List[Union[str, BaseModel]]:
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
    
    result = await handle.remote(request)
    results = result if isinstance(result, list) else [result]
    
    output = []
    for result_item in results:
        text = _extract_text(result_item)
        output.append(_parse_structured_output(text, structured_output) if structured_output else text)
    return output


