"""Deploy multiple models using the VRAM scheduler."""
import ray
import sys
import os

vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vram_scheduler_dir)

import model_orchestrator
import vram_allocator

ModelOrchestrator = model_orchestrator.ModelOrchestrator
get_vram_allocator = vram_allocator.get_vram_allocator

# Model configuration
MODELS = {
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "vram_gb": 2.0,
        "replicas": 1 
    },
    "qwen": {
        "name": "Qwen/Qwen2-0.5B-Instruct",
        "vram_gb": 1.5,  # Model weights ~0.92GB, need room for KV cache
        "replicas": 10  # Multiple replicas can share GPUs (fractional GPU allocation)
    },
}

def main():
    ray.init(address=os.getenv("RAY_ADDRESS", "ray://10.0.1.53:10001"), ignore_reinit_error=True)
    
    get_vram_allocator()
    orchestrator = ModelOrchestrator.remote()
    ray.get(orchestrator.apply.remote(MODELS))
    
    allocator = ray.get_actor("vram_allocator", namespace="system")
    state = ray.get(allocator.get_all_nodes.remote())
    
    print("\nVRAM State:")
    for node_id, info in state.items():
        print(f"  Node {node_id[:8]}: {info['free']:.2f}GB free / {info['total']:.2f}GB total, {len(info['allocs'])} models loaded")

if __name__ == "__main__":
    main()

