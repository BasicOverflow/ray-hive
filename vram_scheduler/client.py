"""Main VRAMScheduler client class."""
import ray
from typing import Dict, Optional
from .core.vram_allocator import get_vram_allocator
from .core.model_orchestrator import ModelOrchestrator
from .shutdown import shutdown_all, shutdown_model
from .utils.ray_utils import init_ray


class VRAMScheduler:
    """Main client for VRAM-aware LLM serving."""
    
    def __init__(self, suppress_logging: bool = True, **kwargs):
        """Initialize VRAMScheduler.
        
        Args:
            suppress_logging: Suppress Ray logging
            **kwargs: Additional Ray init kwargs
        """
        self.suppress_logging = suppress_logging
        self._deployed_models: Dict[str, Dict] = {}
        
        init_ray(suppress_logging=suppress_logging, **kwargs)
        get_vram_allocator()
    
    def deploy_model(
        self,
        model_id: str,
        model_name: str,
        vram_weights_gb: float,
        replicas: Optional[int] = None,
        test_mode: bool = False,
        max_num_seqs: Optional[int] = None,
        max_model_len: int = 8192,
        max_tokens: int = 256,
        hidden_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        dtype: Optional[str] = None,
        max_num_batched_tokens: Optional[int] = None,
        **vllm_kwargs
    ) -> None:
        """Deploy model with VRAM-aware scheduling. One replica per GPU, max replicas capped at available GPUs."""
        max_num_batched_tokens = max_num_batched_tokens or vllm_kwargs.get("max_num_batched_tokens", 2048)
        
        model_configs = {
            model_id: {
                "name": model_name,
                "vram_weights_gb": vram_weights_gb,
                "replicas": replicas,
                "test_mode": test_mode,
                "max_num_seqs": max_num_seqs,
                "max_model_len": max_model_len,
                "max_tokens": max_tokens,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dtype": dtype,
                "max_num_batched_tokens": max_num_batched_tokens,
            }
        }
        
        orchestrator = ModelOrchestrator.remote()
        ray.get(orchestrator.apply.remote(model_configs, {model_id: vllm_kwargs}))
        
        self._deployed_models[model_id] = {
            "model_name": model_name,
            "vram_weights_gb": vram_weights_gb,
            "replicas": replicas,
            "max_num_seqs": max_num_seqs,
            "max_model_len": max_model_len,
            "max_tokens": max_tokens,
        }
    
    def shutdown(self, model_id: Optional[str] = None):
        """Shutdown model deployments. If model_id is None, shutdown all."""
        if model_id is None:
            shutdown_all()
            self._deployed_models.clear()
        else:
            shutdown_model(model_id)
            self._deployed_models.pop(model_id, None)
    
    def get_vram_state(self) -> Dict:
        """Get VRAM state for all GPUs."""
        allocator = get_vram_allocator()
        return ray.get(allocator.get_all_gpus.remote())
    
    def display_vram_state(self):
        """Display current VRAM state in a readable format."""
        state = self.get_vram_state()
        
        print("\nVRAM State (Per-GPU):")
        print("-" * 80)
        for gpu_key, info in sorted(state.items()):
            if len(gpu_key) > 50 or gpu_key.startswith('c'):
                continue
            total = info.get("total", 0)
            free = info.get("free", 0)
            available = info.get("available", 0)
            pending = info.get("pending", 0)
            active = info.get("active", 0)
            active_count = info.get("active_count", 0)
            
            print(f"GPU {gpu_key}:")
            print(f"  Total: {total:.2f}GB, Free: {free:.2f}GB, Available: {available:.2f}GB")
            print(f"  Pending: {pending:.2f}GB, Active: {active:.2f}GB ({active_count} instances)")
        print()

