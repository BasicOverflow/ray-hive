"""
vLLM Model Actor with exact VRAM reservation.

Models declare exact VRAM requirements and the allocator
guarantees safe placement.
"""
import ray
from ray import serve
from typing import Optional

@serve.deployment(
    ray_actor_options={"num_gpus": 0.01},  # Minimal GPU requirement - VRAM allocator handles actual placement
    autoscaling_config={
        "min_replicas": 0,
        "max_replicas": 20,
        "target_num_ongoing_requests_per_replica": 1
    }
)
class VLLMModel:
    """vLLM model with exact VRAM reservation."""
    
    def __init__(self, model_id: str, model_name: str, required_vram_gb: float):
        self.model_id = model_id
        self.required_vram_gb = required_vram_gb
        
        # Get allocator
        allocator = ray.get_actor("vram_allocator", namespace="system")
        
        # Find a node with enough VRAM
        k8s_node_name = ray.get(allocator.find_node_with_vram.remote(required_vram_gb))
        
        if not k8s_node_name:
            raise RuntimeError(
                f"Insufficient VRAM: need {required_vram_gb}GB, "
                f"no node has enough free VRAM"
            )
        
        # HARD admission gate - reserve VRAM on the found node
        reserved = ray.get(allocator.reserve.remote(model_id, k8s_node_name, required_vram_gb))
        
        if not reserved:
            # Race condition - VRAM was taken between find and reserve
            # Ray Serve will retry on another node
            raise RuntimeError(
                f"VRAM reservation failed: need {required_vram_gb}GB on {k8s_node_name}"
            )
        
        self.node_id = k8s_node_name
        
        # Load model - ensure CUDA is available
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot load model.")
        
        from vllm import LLM
        self.llm = LLM(model=model_name)
        
        print(f"Model {model_id} loaded on K8s node {k8s_node_name}, "
              f"reserved {required_vram_gb}GB", flush=True)
    
    def generate(self, prompt: str, **kwargs):
        """Generate text using the model."""
        return self.llm.generate(prompt, **kwargs)
    
    def __del__(self):
        """Release VRAM reservation on cleanup."""
        try:
            allocator = ray.get_actor("vram_allocator", namespace="system")
            ray.get(allocator.release.remote(self.model_id, self.node_id))
        except:
            pass  # Ignore errors during cleanup

