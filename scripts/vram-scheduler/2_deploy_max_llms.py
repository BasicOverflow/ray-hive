#!/usr/bin/env python3
"""
Deploy maximum replicas of a single model to fill available VRAM.

Alternative to 1_deploy_models.py - calculates how many replicas can fit
and deploys that many.

Usage:
    python scripts/vram-scheduler/2_deploy_max_llms.py
    # Or with env vars:
    MODEL_NAME="microsoft/phi-2" MODEL_VRAM_GB=3.0 python scripts/vram-scheduler/2_deploy_max_llms.py
"""
import ray
import os
import sys

# Add vram-scheduler directory to path (folder has hyphen, can't be imported as package)
vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vram_scheduler_dir)

# Import modules directly (since folder name has hyphen)
import vram_allocator
import vllm_model_actor

get_vram_allocator = vram_allocator.get_vram_allocator
VLLMModel = vllm_model_actor.VLLMModel
from ray import serve

RAY_ADDRESS = os.getenv("RAY_ADDRESS", "ray://10.0.1.53:10001")
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/phi-2")
MODEL_VRAM_GB = float(os.getenv("MODEL_VRAM_GB", "3.0"))  # GB per model

# Model configuration
MODELS = {
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "vram_gb": 2.0,
        "replicas": 1 
    },
    "qwen": {
        "name": "Qwen/Qwen2-0.5B-Instruct",
        "vram_gb": 1.0,
        "replicas": 10
    },
}

def main():
    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
    
    # Get allocator
    allocator = get_vram_allocator()
    state = ray.get(allocator.get_all_nodes.remote())
    
    # Calculate total free VRAM across all nodes
    total_free_vram = sum(info.get("free", 0) for info in state.values())
    
    # Calculate max replicas
    max_replicas = int(total_free_vram / MODEL_VRAM_GB)
    
    print(f"Total free VRAM: {total_free_vram:.2f} GB")
    print(f"Model VRAM requirement: {MODEL_VRAM_GB} GB")
    print(f"Can deploy {max_replicas} replicas")
    
    if max_replicas > 0:
        # Deploy with autoscaling up to max replicas
        serve.run(
            VLLMModel.options(
                name="max-llm",
                autoscaling_config={
                    "min_replicas": 0,
                    "max_replicas": max_replicas,
                    "target_num_ongoing_requests_per_replica": 1
                }
            ).bind(
                model_id="max-llm",
                model_name=MODEL_NAME,
                required_vram_gb=MODEL_VRAM_GB
            ),
            name="max-llm",
            route_prefix="/max-llm"
        )
        print(f"✅ Deployed with max {max_replicas} replicas")
    else:
        print("❌ Not enough VRAM to deploy any replicas")

if __name__ == "__main__":
    main()

