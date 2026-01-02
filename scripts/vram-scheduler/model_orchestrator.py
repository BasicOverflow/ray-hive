"""Model Orchestrator - declarative model deployment."""
import ray
from ray import serve
from typing import Dict
import sys
import os
import time

vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
if vram_scheduler_dir not in sys.path:
    sys.path.insert(0, vram_scheduler_dir)

import vllm_model_actor
VLLMModel = vllm_model_actor.VLLMModel

@ray.remote(num_cpus=0)
class ModelOrchestrator:
    """Deploys models via Ray Serve based on configuration."""
    
    def apply(self, model_configs: Dict):
        """Deploy all models from configuration."""
        for model_id, config in model_configs.items():
            print(f"Deploying {model_id}: {config['name']}, {config['vram_gb']}GB, {config['replicas']} replicas")
            
            serve.run(
                VLLMModel.options(
                    name=model_id,
                    autoscaling_config={
                        "min_replicas": config["replicas"],
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
            time.sleep(0.5)  # Stagger deployments to reduce race conditions
        
        print("All models deployed!")

