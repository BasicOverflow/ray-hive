"""
Model Orchestrator - declarative model deployment.

Defines desired model state and deploys them via Ray Serve.
"""
import ray
from ray import serve
from typing import Dict

# Declarative model state
MODELS = {
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B",
        "vram_gb": 2.0,
        "replicas": 10
    },
    "phi2": {
        "name": "microsoft/phi-2",
        "vram_gb": 3.0,
        "replicas": 8
    },
    "qwen": {
        "name": "Qwen/Qwen2-0.5B",
        "vram_gb": 1.0,
        "replicas": 12
    },
}

@ray.remote(num_cpus=0)
class ModelOrchestrator:
    """Orchestrates model deployment based on declarative state."""
    
    def apply(self, model_configs: Dict):
        """Deploy models according to configuration."""
        import sys
        import os
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, repo_root)
        from scripts.vram_scheduler.vllm_model_actor import VLLMModel
        
        for model_id, config in model_configs.items():
            print(f"Deploying {model_id}: {config['name']}, "
                  f"{config['vram_gb']}GB, {config['replicas']} replicas")
            
            serve.run(
                VLLMModel.options(
                    name=model_id,
                    autoscaling_config={
                        "min_replicas": 0,
                        "max_replicas": config["replicas"]
                    }
                ).bind(
                    model_id=model_id,
                    model_name=config["name"],
                    required_vram_gb=config["vram_gb"]
                ),
                name=model_id,
                route_prefix=f"/{model_id}"
            )
        
        print("All models deployed!")


if __name__ == "__main__":
    import sys
    import os
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, scripts_dir)
    
    ray.init(address="ray://10.0.1.53:10001", ignore_reinit_error=True)
    
    # Initialize allocator
    from vram_scheduler.vram_allocator import get_vram_allocator
    get_vram_allocator()
    
    # Deploy models
    orchestrator = ModelOrchestrator.remote()
    ray.get(orchestrator.apply.remote(MODELS))

