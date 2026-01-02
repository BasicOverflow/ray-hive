"""Deploy maximum replicas of a single model to fill available VRAM."""
import ray
import os
import sys
from ray import serve

vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vram_scheduler_dir)

import vram_allocator
import vllm_model_actor

get_vram_allocator = vram_allocator.get_vram_allocator
VLLMModel = vllm_model_actor.VLLMModel

MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/phi-2")
MODEL_VRAM_GB = float(os.getenv("MODEL_VRAM_GB", "3.0"))

def main():
    ray.init(address=os.getenv("RAY_ADDRESS", "ray://10.0.1.53:10001"), ignore_reinit_error=True)
    
    # Calculate max replicas based on available VRAM
    allocator = get_vram_allocator()
    state = ray.get(allocator.get_all_nodes.remote())
    total_free_vram = sum(info.get("free", 0) for info in state.values())
    max_replicas = int(total_free_vram / MODEL_VRAM_GB)
    
    print(f"Total free VRAM: {total_free_vram:.2f} GB")
    print(f"Model VRAM requirement: {MODEL_VRAM_GB} GB")
    print(f"Can deploy {max_replicas} replicas")
    
    if max_replicas > 0:
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

