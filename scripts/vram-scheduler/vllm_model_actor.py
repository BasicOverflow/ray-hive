"""vLLM Model Actor - Per-GPU sequential initialization."""
import ray
from ray import serve

@serve.deployment(
    ray_actor_options={
        "num_gpus": 0.01,  # Fractional GPU - allows multiple replicas per GPU
        "memory": 2 * 1024 * 1024 * 1024,  # Request 2GB system RAM per replica
    },
    autoscaling_config={
        "min_replicas": 0,
        "max_replicas": 100,
        "target_num_ongoing_requests_per_replica": 1
    }
)
class VLLMModel:
    """vLLM model with per-GPU sequential initialization."""
    
    def __init__(self, model_id: str, model_name: str, required_vram_gb: float, target_gpu_id: str = None):
        self.model_id = model_id
        self.required_vram_gb = required_vram_gb
        
        # Set up Python path first
        import sys
        import os
        
        # Determine GPU key - target_gpu_id is always provided now
        if target_gpu_id is None:
            raise RuntimeError("target_gpu_id must be provided")
        
        # Set CUDA_VISIBLE_DEVICES BEFORE importing torch
        # This forces the actor to use only the specified GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu_id
        
        vllm_paths = [
            "/vllm-install",
            "/vllm-install/lib/python3.12/site-packages"
        ]
        for path in vllm_paths:
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)
        
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        
        # Get VRAM allocator
        try:
            allocator = ray.get_actor("vram_allocator", namespace="system")
        except ValueError:
            raise RuntimeError("VRAM allocator actor doesn't exist. Run deployment script first.")
        
        # Get unique replica ID
        actor_id = ray.get_runtime_context().get_actor_id()
        self.replica_id = f"{model_id}-{actor_id}"
        
        # Get K8s node name from hostname
        import socket
        hostname = socket.gethostname()
        parts = hostname.split("-")
        if len(parts) >= 4 and parts[0] == "raycluster" and parts[1] == "gpu" and parts[2] == "workers":
            worker_idx = None
            for i in range(len(parts)):
                if parts[i] == "worker":
                    worker_idx = i
                    break
            if worker_idx and worker_idx > 2:
                k8s_node_name = "-".join(parts[3:worker_idx])
            else:
                k8s_node_name = "-".join(parts[3:-1]) if parts[-1] != "worker" else "-".join(parts[3:-2])
        else:
            k8s_node_name = hostname
        
        gpu_key = f"{k8s_node_name}:gpu{target_gpu_id}"
        self.gpu_key = gpu_key
        
        # Acquire per-GPU lock on the actual GPU BEFORE any memory checks or vLLM initialization
        import time
        max_lock_attempts = 300
        for attempt in range(max_lock_attempts):
            if ray.get(allocator.acquire_gpu_lock.remote(gpu_key, self.replica_id)):
                break
            if attempt % 20 == 0:
                print(f"[{model_id}] Waiting for GPU lock on {gpu_key} (attempt {attempt + 1}/{max_lock_attempts})...", flush=True)
            time.sleep(0.5)
        else:
            raise RuntimeError(f"Failed to acquire GPU lock on {gpu_key} after {max_lock_attempts} attempts")
        
        try:
            # Reserve VRAM on the actual GPU
            reserved = ray.get(allocator.reserve.remote(self.replica_id, gpu_key, required_vram_gb))
            if not reserved:
                gpu_info = ray.get(allocator.get_gpu_vram.remote(gpu_key))
                available_gb = gpu_info.get("available", 0) if gpu_info else 0
                raise RuntimeError(f"VRAM reservation failed on {gpu_key}: need {required_vram_gb:.2f}GB, have {available_gb:.2f}GB available")
            
            # Get GPU memory info from allocator (uses nvidia-smi data from daemonset)
            gpu_info = ray.get(allocator.get_gpu_vram.remote(gpu_key))
            if not gpu_info:
                raise RuntimeError(f"Could not get GPU VRAM info for {gpu_key}")
            
            # Use total from allocator (nvidia-smi data from daemonset)
            total_memory_gb = gpu_info.get("total", 0)
            if total_memory_gb == 0:
                # Fallback to CUDA if allocator doesn't have total
                cuda_device_id = torch.cuda.current_device()
                total_memory_gb = torch.cuda.get_device_properties(cuda_device_id).total_memory / (1024**3)
            
            # Calculate utilization based on actual VRAM requirement
            gpu_memory_utilization = required_vram_gb / total_memory_gb
            gpu_memory_utilization = max(0.05, min(gpu_memory_utilization, 1.0))  # Cap at 100%
            
            # Import vLLM
            try:
                from vllm import LLM
            except ImportError as e:
                raise RuntimeError(f"Failed to import vllm: {e}") from e
            
            # Initialize vLLM with simple retry for rate limits only
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    self.llm = LLM(
                        model=model_name,
                        gpu_memory_utilization=gpu_memory_utilization,
                        max_model_len=8192,
                        enforce_eager=True,
                        max_num_seqs=1,
                        swap_space=0,
                        enable_chunked_prefill=False,
                    )
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    if "rate limit" in error_str or "429" in error_str:
                        if attempt < max_retries - 1:
                            import random
                            wait_time = (2 ** attempt) * 10 + random.uniform(0, 5)
                            print(f"[{model_id}] Rate limit, retrying in {wait_time:.1f}s...", flush=True)
                            time.sleep(wait_time)
                            continue
                    if attempt == max_retries - 1:
                        raise
            
            # Mark as successfully initialized
            ray.get(allocator.mark_initialized.remote(self.replica_id, gpu_key))
            print(f"[{model_id}] ✅ Model loaded on {gpu_key}", flush=True)
            
        except Exception as e:
            # Release reservation and lock on failure
            ray.get(allocator.release.remote(self.replica_id, gpu_key))
            raise
        finally:
            # Always release lock
            ray.get(allocator.release_gpu_lock.remote(gpu_key, self.replica_id))
    
    def __del__(self):
        """Cleanup: release VRAM reservation."""
        if self.replica_id and self.gpu_key:
            try:
                allocator = ray.get_actor("vram_allocator", namespace="system")
                ray.get(allocator.release.remote(self.replica_id, self.gpu_key))
            except:
                pass
        
    
    def generate(self, prompt: str, **kwargs):
        """Generate text using the model."""
        from vllm import SamplingParams
        
        # Extract sampling parameters from kwargs
        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", 256),
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            stop=kwargs.get("stop", None),
            stop_token_ids=kwargs.get("stop_token_ids", None),
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        # Extract text from vLLM RequestOutput objects
        if hasattr(outputs, '__iter__'):
            results = []
            for output in outputs:
                if hasattr(output, 'outputs'):
                    for out in output.outputs:
                        if hasattr(out, 'text'):
                            results.append(out.text)
                elif hasattr(output, 'text'):
                    results.append(output.text)
            return results if results else [str(outputs)]
        return [str(outputs)]
