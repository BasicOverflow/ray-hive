"""
RayHive client — high-level API for deploying models and querying VRAM state.

Wraps DeployService (serialized deploy) and gpu_registry (VRAM tracking).
"""
import time

import ray
from typing import Dict, List, Optional, Type, Union

from .core.deployment import get_deploy_service
from .core.gpu_alloc import BaseGpuAllocator, reject_unsupported_host_ram_kwargs
from .core.gpu_registry import get_gpu_registry
from .core.model_specs.attention import BaseAttentionSpecs
from .core.model_specs.estimate import load_hf_config_dict
from .core.model_specs.planner import normalize_hf_config
from .core.ray_utils import init_ray, shutdown_all, shutdown_model, suppress_ray_warnings
from .core.ray_utils.display import (
    print_banner,
    print_deployment_plan,
    warn,
)
from .core.ray_utils.placement import plan_replica_groups

# Lifted out of vllm_kwargs into planner config (must not also reach engine_kwargs).
_PLANNER_VLLM_KEYS = ("max_num_seqs", "max_num_batched_tokens")


def _split_vllm_kwargs(vllm_kwargs: dict | None) -> tuple[dict, dict]:
    """Pop planner overrides from vllm_kwargs; return (overrides, remaining)."""
    remaining = dict(vllm_kwargs or {})
    overrides = {}
    for key in _PLANNER_VLLM_KEYS:
        if key in remaining:
            overrides[key] = remaining.pop(key)
    return overrides, remaining


def _retry_if_registry_empty(fn):
    """DaemonSet needs a beat after kill_gpu_registry; one retry like re-running the script."""
    try:
        return fn()
    except Exception as e:
        if "not in registry" not in str(e) and "Known: []" not in str(e):
            raise
        time.sleep(5)
        return fn()


class RayHive:
    """Main client for distributed LLM serving."""

    def __init__(self, address: str, suppress_logging: bool = True, show_banner: bool = True, **kwargs):
        """Connect to Ray cluster at address and ensure singleton actors exist."""
        if show_banner:
            print_banner()
        warn(
            "WARNING: vLLM has model-family usage guides — check them before deploy "
            "so you pass the right HF config / vllm kwargs (quantization, rope, etc.). "
            "Those guides also recommend env vars per GPU architecture for better performance."
        )
        suppress_ray_warnings(suppress_logging)
        init_ray(address, suppress_logging=suppress_logging, **kwargs)
        get_gpu_registry()


    def deploy_model(
        self,
        model_id: str,
        model_name: str,
        max_input_prompt_length: int,
        max_output_prompt_length: int,
        replicas: int = 1,
        gpu: Optional[Union[str, List[str]]] = None,
        cpu_ram_per_instance: float = 0,
        attention_cls: Optional[Type[BaseAttentionSpecs]] = None,
        allocation_cls: Optional[Type[BaseGpuAllocator]] = None,
        idle_timeout: int = -1,
        sleep_timeout: int = -1,
        vllm_kwargs: Optional[dict] = None,
    ) -> dict:
        """
        Deploy a model with VRAM-aware scheduling.

        Returns when the model is ready (router up + warmed). Status dict includes
        model_id, status="ready", route, and per-replica plan/GPU info.

        max_input_prompt_length and max_output_prompt_length are required.
        Pass planner overrides (max_num_seqs, max_num_batched_tokens) inside
        vllm_kwargs — they are lifted automatically.
        replicas=-1 deploys to all eligible GPUs (or all eligible TP groups when auto TP>1).
        attention_cls defaults to BaseAttentionSpecs (standard transformer KV sizing).
        allocation_cls defaults to RayPerformanceAllocator for single-GPU auto placement;
        ignored when gpu= is set. Auto TP packing always uses RayTensorParallelAllocator.
        gpu=None: place on one GPU if any fits; otherwise same-node TP (2, 3, ...).
        gpu=[a,b,...] + replicas=1: one same-node TP group.
        gpu=[a,b,...] + replicas=len(list): N single-GPU pins (one replica per GPU).
        cpu_ram_per_instance: hive host-RAM extension (not a vLLM arg) — TP=1 only. 0 off;
          -1 = auto weight-spill need (capped at 70% of Ray free host memory / replicas
          on host); >0 hard GiB ceiling per replica. Weights stay on GPU if they fit;
          overflow spills to host. Leftover host RAM is unused (no CPU KV connector).
        sleep_timeout: seconds of inactivity before all replicas sleep (level 1); -1 never.
        idle_timeout: seconds of inactivity before full self-shutdown; -1 never.
        When both are set, idle_timeout must be greater than sleep_timeout.
        """
        if idle_timeout != -1 and idle_timeout <= 0:
            raise ValueError("idle_timeout must be -1 (never) or a positive number of seconds")
        if sleep_timeout != -1 and sleep_timeout <= 0:
            raise ValueError("sleep_timeout must be -1 (never) or a positive number of seconds")
        if idle_timeout > 0 and sleep_timeout > 0 and idle_timeout <= sleep_timeout:
            raise ValueError("idle_timeout must be greater than sleep_timeout when both are set")
        planner_overrides, vllm_kwargs = _split_vllm_kwargs(vllm_kwargs)
        reject_unsupported_host_ram_kwargs(vllm_kwargs)

        config = {
            "name": model_name,
            "replicas": replicas,
            "gpu": gpu,
            "max_input_prompt_length": max_input_prompt_length,
            "max_output_prompt_length": max_output_prompt_length,
            "cpu_ram_per_instance": cpu_ram_per_instance,
            "attention_cls": attention_cls,
            "allocation_cls": allocation_cls,
            "idle_timeout": idle_timeout,
            "sleep_timeout": sleep_timeout,
            **planner_overrides,
        }

        deploy_svc = get_deploy_service()

        def _deploy():
            return ray.get(deploy_svc.deploy_models.remote(
                model_configs={model_id: config},
                vllm_kwargs={model_id: vllm_kwargs},
            ))

        replicas_info = _retry_if_registry_empty(_deploy)[model_id]
        return {
            "model_id": model_id,
            "status": "ready",
            "route": f"/{model_id}",
            "replicas": replicas_info,
        }


    def estimate_vram(
        self,
        model_name: str,
        max_input_prompt_length: int,
        max_output_prompt_length: int,
        replicas: int = 1,
        gpu: Optional[Union[str, List[str]]] = None,
        cpu_ram_per_instance: float = 0,
        attention_cls: Optional[Type[BaseAttentionSpecs]] = None,
        allocation_cls: Optional[Type[BaseGpuAllocator]] = None,
        vllm_kwargs: Optional[dict] = None,
    ) -> dict:
        """Dry-run packing against live GPUs; print the same plan deploy would use."""
        planner_overrides, vllm_kwargs = _split_vllm_kwargs(vllm_kwargs)
        reject_unsupported_host_ram_kwargs(vllm_kwargs)
        vllm_kwargs.pop("default_chat_template_kwargs", None)
        vllm_kwargs.pop("tensor_parallel_size", None)
        vllm_kwargs.pop("distributed_executor_backend", None)
        vllm_kwargs.pop("idle_timeout", None)
        vllm_kwargs.pop("sleep_timeout", None)

        config = {
            "name": model_name,
            "replicas": replicas,
            "gpu": gpu,
            "max_input_prompt_length": max_input_prompt_length,
            "max_output_prompt_length": max_output_prompt_length,
            "cpu_ram_per_instance": cpu_ram_per_instance,
            "attention_cls": attention_cls,
            "allocation_cls": allocation_cls,
            **planner_overrides,
        }

        hf_params = normalize_hf_config(load_hf_config_dict(model_name))

        def _plan():
            return plan_replica_groups(
                self.get_vram_state(), config, hf_params, vllm_kwargs, model_id=model_name
            )

        results = _retry_if_registry_empty(_plan)
        print_deployment_plan(model_name, results)
        return results


    def shutdown(self, model_id: Optional[str] = None):
        """Shutdown one model or all models (model_id=None)."""
        if model_id is None:
            shutdown_all()
        else:
            shutdown_model(model_id)


    def get_vram_state(self) -> Dict:
        """Return live VRAM state dict for all GPUs from the registry."""
        registry = get_gpu_registry()
        return ray.get(registry.get_all_gpus.remote())
