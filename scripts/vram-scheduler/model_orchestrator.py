"""
Model Orchestrator - declarative model deployment.

Defines desired model state and deploys them via Ray Serve.
"""
import ray
from ray import serve
from typing import Dict
import sys
import os

# Set up path for imports (folder has hyphen, can't be imported as package)
vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
if vram_scheduler_dir not in sys.path:
    sys.path.insert(0, vram_scheduler_dir)

# Import at module level so it's available when serialized to remote workers
import vllm_model_actor
VLLMModel = vllm_model_actor.VLLMModel

@ray.remote(num_cpus=0)
class ModelOrchestrator:
    """Orchestrates model deployment based on declarative state."""
    
    def apply(self, model_configs: Dict):
        """Deploy models according to configuration."""
        
        for model_id, config in model_configs.items():
            print(f"Deploying {model_id}: {config['name']}, "
                  f"{config['vram_gb']}GB, {config['replicas']} replicas")
            
            serve.run(
                VLLMModel.options(
                    name=model_id,
                    autoscaling_config={
                        "min_replicas": config["replicas"],  # Target replicas - VRAM allocator gates actual placement
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

