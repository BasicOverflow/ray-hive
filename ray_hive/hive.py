"""
RayHive client — high-level API for deploying models and querying VRAM state.

Wraps DeployService (serialized deploy/shutdown) and gpu_registry (VRAM tracking).
"""
import ray
from typing import Dict, List, Optional, Type, Union

from .core.deployment import get_deploy_service
from .core.gpu_alloc import BaseGpuAllocator
from .core.gpu_registry import get_gpu_registry
from .core.model_specs.attention import BaseAttentionSpecs
from .core.ray_utils import init_ray, shutdown_all, shutdown_model, suppress_ray_warnings


class RayHive:
    """Main client for distributed LLM serving."""

    def __init__(self, address: str, suppress_logging: bool = True, **kwargs):
        """Connect to Ray cluster at address and ensure singleton actors exist."""
        print(
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
        max_num_seqs: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        cpu_ram_per_instance: float = 0,
        attention_cls: Optional[Type[BaseAttentionSpecs]] = None,
        allocation_cls: Optional[Type[BaseGpuAllocator]] = None,
        idle_timeout: int = -1,
        **vllm_kwargs
    ) -> None:
        """
        Deploy a model with VRAM-aware scheduling.

        max_input_prompt_length and max_output_prompt_length are required.
        max_num_seqs and max_num_batched_tokens are estimated from VRAM unless overridden.
        replicas=-1 deploys to all eligible GPUs (or all eligible TP groups when auto TP>1).
        attention_cls defaults to BaseAttentionSpecs (standard transformer KV sizing).
        allocation_cls defaults to RayPerformanceAllocator for single-GPU auto placement;
        ignored when gpu= is set. Auto TP packing always uses RayTensorParallelAllocator.
        gpu=None: place on one GPU if any fits; otherwise same-node TP (2, 3, ...).
        gpu=[a,b,...] + replicas=1: one same-node TP group.
        gpu=[a,b,...] + replicas=len(list): N single-GPU pins (one replica per GPU).
        cpu_ram_per_instance: sole host-RAM extension arg — 0 off; -1 = 85% free VRAM;
          >0 hard GiB. Weights stay on GPU if they fit; overflow spills to host; leftover → KV.
        """
        if idle_timeout != -1 and idle_timeout <= 0:
            raise ValueError("idle_timeout must be -1 (never) or a positive number of seconds")

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
        }
        if max_num_seqs is not None:
            config["max_num_seqs"] = max_num_seqs
        if max_num_batched_tokens is not None:
            config["max_num_batched_tokens"] = max_num_batched_tokens

        deploy_svc = get_deploy_service()
        results = ray.get(deploy_svc.deploy_models.remote(
            model_configs={model_id: config},
            vllm_kwargs={model_id: vllm_kwargs},
        ))

        if model_id in results and results[model_id]:
            print(f"\n{'='*80}")
            print(f"Deployment Plan Summary: {model_id}")
            print(f"{'='*80}")
            for replica_id, summary in results[model_id].items():
                plan = summary["plan"]
                gpu_keys = summary.get("gpu_keys") or [summary["gpu_key"]]
                tp = plan.get("tensor_parallel_size", len(gpu_keys))
                print(f"\nReplica: {replica_id}")
                print(f"  GPU(s): {', '.join(gpu_keys)}")
                print(f"  tensor_parallel_size: {tp}")
                print(f"  max_num_seqs: {plan['max_num_seqs']}")
                print(f"  max_num_batched_tokens: {plan['max_num_batched_tokens']}")
                print(f"  gpu_memory_utilization: {plan['gpu_memory_utilization']:.3f}")
                print(f"  total_vram_gb (per GPU): {plan['total_vram_gb']:.2f}")
                print(f"  cpu_ram_budget_gb: {plan.get('cpu_ram_budget_gb', 0):.2f}")
                print(f"  cpu_kv_offload_gb: {plan.get('cpu_kv_offload_gb', 0):.2f}")
                print(f"  cpu_offload_gb (weights): {plan.get('cpu_offload_gb', 0):.2f}")
            print(f"{'='*80}\n")


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
