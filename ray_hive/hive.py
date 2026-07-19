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
from .ray_utils import init_ray, suppress_ray_warnings
from .shutdown import shutdown_all, shutdown_model


class RayHive:
    """Main client for distributed LLM serving."""

    def __init__(self, suppress_logging: bool = True, **kwargs):
        """Connect to Ray cluster and ensure singleton actors exist."""
        print(
            "WARNING: vLLM has model-family usage guides — check them before deploy "
            "so you pass the right HF config / vllm kwargs (quantization, rope, etc.). "
            "Those guides also recommend env vars per GPU architecture for better performance."
        )
        suppress_ray_warnings(suppress_logging)
        init_ray(suppress_logging=suppress_logging, **kwargs)
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
        swap_space_per_instance: int = 3,
        attention_cls: Optional[Type[BaseAttentionSpecs]] = None,
        allocation_cls: Optional[Type[BaseGpuAllocator]] = None,
        **vllm_kwargs
    ) -> None:
        """
        Deploy a model with VRAM-aware scheduling.

        max_input_prompt_length and max_output_prompt_length are required.
        max_num_seqs and max_num_batched_tokens are estimated from VRAM unless overridden.
        replicas=-1 deploys to all eligible GPUs.
        attention_cls defaults to BaseAttentionSpecs (standard transformer KV sizing).
        allocation_cls defaults to RayPerformanceAllocator; ignored when gpu= is set.
        """
        config = {
            "name": model_name,
            "replicas": replicas,
            "gpu": gpu,
            "max_input_prompt_length": max_input_prompt_length,
            "max_output_prompt_length": max_output_prompt_length,
            "swap_space_per_instance": swap_space_per_instance,
            "attention_cls": attention_cls,
            "allocation_cls": allocation_cls,
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
                gpu_key = summary["gpu_key"]
                print(f"\nReplica: {replica_id}")
                print(f"  GPU: {gpu_key}")
                print(f"  max_num_seqs: {plan['max_num_seqs']}")
                print(f"  max_num_batched_tokens: {plan['max_num_batched_tokens']}")
                print(f"  gpu_memory_utilization: {plan['gpu_memory_utilization']:.3f}")
                print(f"  total_vram_gb: {plan['total_vram_gb']:.2f}")
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


    def display_vram_state(self):
        """Print available/total VRAM for each GPU."""
        state = self.get_vram_state()
        for gpu_key, info in sorted(state.items()):
            print(f"GPU {gpu_key}: {info.get('available', 0):.2f}GB available / {info.get('total', 0):.2f}GB total")
