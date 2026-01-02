"""vLLM Model Actor with exact VRAM reservation."""
import ray
from ray import serve

@serve.deployment(
    ray_actor_options={"num_gpus": 0.01},  # Fractional GPU - allows multiple replicas per GPU, memory slicing handles limits
    autoscaling_config={
        "min_replicas": 0,
        "max_replicas": 20,
        "target_num_ongoing_requests_per_replica": 1
    }
)
class VLLMModel:
    """vLLM model with exact VRAM reservation using CUDA memory slicing."""
    
    def __init__(self, model_id: str, model_name: str, required_vram_gb: float):
        self.model_id = model_id
        self.required_vram_gb = required_vram_gb
        
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot load model.")
        
        # Set memory fraction before vLLM import (subprocesses inherit it)
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        memory_with_buffer = required_vram_gb * 1.7  # 70% buffer for KV cache/overhead
        memory_fraction = min(memory_with_buffer / total_gpu_memory, 0.90)
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
        
        # Find node with available VRAM and reserve it
        allocator = ray.get_actor("vram_allocator", namespace="system")
        k8s_node_name = ray.get(allocator.find_node_with_vram.remote(required_vram_gb))
        
        if not k8s_node_name:
            raise RuntimeError(f"Insufficient VRAM: need {required_vram_gb}GB")
        
        reserved = ray.get(allocator.reserve.remote(model_id, k8s_node_name, required_vram_gb))
        if not reserved:
            raise RuntimeError(f"VRAM reservation failed: need {required_vram_gb}GB on {k8s_node_name}")
        
        self.node_id = k8s_node_name
        
        from vllm import LLM
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=memory_fraction,
            enforce_eager=True,
            max_num_seqs=1,
            swap_space=0,
            enable_chunked_prefill=False,
        )
        
        print(f"Model {model_id} loaded on {k8s_node_name}, reserved {required_vram_gb}GB", flush=True)
    
    def generate(self, prompt: str, **kwargs):
        """Generate text using the model. Returns extracted text strings."""
        outputs = self.llm.generate(prompt, **kwargs)
        # Extract text from vLLM RequestOutput objects to avoid serialization issues
        # vLLM returns list of RequestOutput, each with outputs[0].text
        if not isinstance(outputs, list):
            outputs = [outputs]
        return [output.outputs[0].text for output in outputs]
    
    def __del__(self):
        """Release VRAM reservation on cleanup."""
        try:
            allocator = ray.get_actor("vram_allocator", namespace="system")
            ray.get(allocator.release.remote(self.model_id, self.node_id))
        except:
            pass

