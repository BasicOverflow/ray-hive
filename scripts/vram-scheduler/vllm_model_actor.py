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
        self.replica_id = None
        self.gpu_key = None
        
        # Set up Python path first
        import sys
        import os
        
        # Set CUDA_VISIBLE_DEVICES BEFORE importing torch if target_gpu_id is specified
        # This forces the actor to use only the specified GPU
        if target_gpu_id is not None:
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
        replica_id = f"{model_id}-{actor_id}"
        self.replica_id = replica_id
        
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
        
        # Use VRAM requirement directly (already accounts for model + KV cache + overhead)
        actual_required_gb = required_vram_gb
        
        # Detect actual GPU being used (CUDA is already initialized)
        actual_gpu_id = torch.cuda.current_device()
        actual_gpu_key = f"{k8s_node_name}:gpu{actual_gpu_id}"
        
        # Try to find an available GPU on this node first
        # This handles cases where Ray schedules us on a node but we need to find the right GPU
        found_gpu = ray.get(allocator.find_gpu_with_vram.remote(actual_required_gb, node_id=k8s_node_name))
        
        if found_gpu and found_gpu.startswith(k8s_node_name + ":"):
            # Found an available GPU on this node
            gpu_key = found_gpu
        else:
            # No available GPU on this node, use the one we're actually on
            # This will fail during reservation if it's full, causing Ray to reschedule
            gpu_key = actual_gpu_key
        
        self.gpu_key = gpu_key
        
        # Acquire per-GPU lock on the actual GPU BEFORE any memory checks or vLLM initialization
        import time
        max_lock_attempts = 300
        lock_acquired = False
        for attempt in range(max_lock_attempts):
            lock_acquired = ray.get(allocator.acquire_gpu_lock.remote(gpu_key, replica_id))
            if lock_acquired:
                break
            if attempt % 20 == 0:
                print(f"[{model_id}] Waiting for GPU lock on {gpu_key} (attempt {attempt + 1}/{max_lock_attempts})...", flush=True)
            time.sleep(0.5)
        
        if not lock_acquired:
            raise RuntimeError(f"Failed to acquire GPU lock on {gpu_key} after {max_lock_attempts} attempts")
        
        try:
            # Reserve VRAM on the actual GPU
            reserved = ray.get(allocator.reserve.remote(replica_id, gpu_key, actual_required_gb))
            if not reserved:
                gpu_info = ray.get(allocator.get_gpu_vram.remote(gpu_key))
                available_gb = gpu_info.get("available", 0) if gpu_info else 0
                raise RuntimeError(f"VRAM reservation failed on {gpu_key}: need {actual_required_gb:.2f}GB, have {available_gb:.2f}GB available")
            
            # Get GPU memory info from allocator (uses nvidia-smi data from daemonset)
            gpu_info = ray.get(allocator.get_gpu_vram.remote(gpu_key))
            if not gpu_info:
                raise RuntimeError(f"Could not get GPU VRAM info for {gpu_key}")
            
            # Use total from allocator (nvidia-smi data from daemonset)
            total_memory_gb = gpu_info.get("total", 0)
            if total_memory_gb == 0:
                # Fallback to CUDA if allocator doesn't have total
                total_memory_gb = torch.cuda.get_device_properties(actual_gpu_id).total_memory / (1024**3)
            
            # Calculate utilization based on actual VRAM requirement (no conservative multipliers)
            gpu_memory_utilization = actual_required_gb / total_memory_gb
            gpu_memory_utilization = max(0.05, min(gpu_memory_utilization, 0.95))  # Cap at 95% to avoid OOM
            
            # Memory check using daemonset's nvidia-smi data (consistent with reservation system)
            # The reservation already validated availability, but we double-check using the same source
            free_gb = gpu_info.get("free", 0)  # nvidia-smi free from daemonset
            requested_by_vllm_gb = gpu_memory_utilization * total_memory_gb
            
            if free_gb < requested_by_vllm_gb:
                available_gb = gpu_info.get("available", 0)
                raise RuntimeError(f"Insufficient GPU memory on {gpu_key}: need {requested_by_vllm_gb:.2f}GB, have {free_gb:.2f}GB free (available: {available_gb:.2f}GB) from nvidia-smi")
            
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
            ray.get(allocator.mark_initialized.remote(replica_id, gpu_key))
            print(f"[{model_id}] ✅ Model loaded on {gpu_key}", flush=True)
            
        except Exception as e:
            # Release reservation and lock on failure
            ray.get(allocator.release.remote(replica_id, gpu_key))
            raise
        finally:
            # Always release lock
            ray.get(allocator.release_gpu_lock.remote(gpu_key, replica_id))
    
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
        outputs = self.llm.generate(prompt, **kwargs)
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
