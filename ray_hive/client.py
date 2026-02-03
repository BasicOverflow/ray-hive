# Main RayHive client class
# Provides high-level API for deploying models and managing VRAM state
# Wraps ModelOrchestrator and VRAMAllocator for user-friendly interface

import ray
from typing import Dict, Optional, Union
from .core.vram_allocator import get_vram_allocator
from .core.model_orchestrator import ModelOrchestrator
from .shutdown import shutdown_all, shutdown_model
from .utils.ray_utils import init_ray, suppress_ray_warnings


class RayHive:
    """Main client for distributed LLM serving."""
    
    def __init__(self, suppress_logging: bool = True, **kwargs):
        """Initialize RayHive.
        
        Args:
            suppress_logging: Suppress Ray logging
            **kwargs: Additional Ray init kwargs
        """
        suppress_ray_warnings(suppress_logging)
        init_ray(suppress_logging=suppress_logging, **kwargs)
        get_vram_allocator()
    
    def deploy_model(
        self,
        model_id: str,
        model_name: str,
        vram_weights_gb: float,
        max_input_prompt_length: int,
        max_output_prompt_length: int,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        replicas: Optional[int] = None,
        test_mode: bool = False,
        test_gpu: Optional[str] = None,
        gpu_utilization_target: float = 0.96,
        swap_space_per_instance: Union[int, str] = 0,
        **vllm_kwargs
    ) -> None:
        """Deploy model with VRAM-aware scheduling. One replica per GPU, max replicas possible capped at available GPUs.
        
        Args:
            max_input_prompt_length: Maximum input prompt length
            max_output_prompt_length: Maximum output prompt length
            max_num_seqs: Maximum number of concurrent sequences
            max_num_batched_tokens: Maximum number of batched tokens
            test_gpu: When test_mode=True, specify GPU to deploy to (e.g., "ergos-06-nv:gpu0")
            gpu_utilization_target: GPU utilization target (default 0.96)
            swap_space_per_instance: Swap space per instance in GB, or 'max' to use all available CPU RAM divided evenly across GPUs
        """
        model_configs = {
            model_id: {
                "name": model_name,
                "vram_weights_gb": vram_weights_gb,
                "replicas": replicas,
                "test_mode": test_mode,
                "test_gpu": test_gpu,
                "max_input_prompt_length": max_input_prompt_length,
                "max_output_prompt_length": max_output_prompt_length,
                "max_num_seqs": max_num_seqs,
                "max_num_batched_tokens": max_num_batched_tokens,
                "gpu_utilization_target": gpu_utilization_target,
                "swap_space_per_instance": swap_space_per_instance,
            }
        }
        
        orchestrator = ModelOrchestrator.remote()
        results = ray.get(orchestrator.apply.remote(model_configs, {model_id: vllm_kwargs}))
        
        if model_id in results and results[model_id]:
            print(f"\n{'='*80}")
            print(f"Deployment Calculation Summary: {model_id}")
            print(f"{'='*80}")
            for _, summary in results[model_id].items():
                calc = summary['calc']
                gpu_key = summary['gpu_key']
                print(f"\nGPU: {gpu_key}")
                print(f"  Model: {calc['model_name']}")
                print(f"  GPU Model: {calc['gpu_model']} ({calc['total_vram_gb']:.2f} GB)")
                print(f"  max_num_seqs: {calc['max_num_seqs']}")
                print(f"  max_num_batched_tokens: {calc['max_num_batched_tokens']}")
                swap_space = calc['swap_space_gb']
                if swap_space and swap_space > 0:
                    print(f"  CPU Offloading: Enabled ({swap_space:.2f} GB swap)")
                    print(f"    Active Seqs (GPU): {calc['active_seqs_gpu']}")
                    print(f"    Parked Seqs (CPU): {calc['parked_seqs_cpu']}")
                    print(f"    Total Concurrent: {calc['total_concurrent_seqs']}")
                else:
                    print(f"  CPU Offloading: Disabled")
            print(f"{'='*80}\n")
    
    def shutdown(self, model_id: Optional[str] = None):
        if model_id is None:
            shutdown_all()
        else:
            shutdown_model(model_id)
    
    def get_vram_state(self) -> Dict:
        """Get VRAM state for all GPUs."""
        allocator = get_vram_allocator()
        return ray.get(allocator.get_all_gpus.remote())
    
    def display_vram_state(self):
        state = self.get_vram_state()
        for gpu_key, info in sorted(state.items()):
            if len(gpu_key) > 50 or gpu_key.startswith('c'):
                continue
            print(f"GPU {gpu_key}: {info.get('available', 0):.2f}GB available / {info.get('total', 0):.2f}GB total")
    

