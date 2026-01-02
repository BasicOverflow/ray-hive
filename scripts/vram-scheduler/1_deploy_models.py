"""
Deploy multiple models using the VRAM scheduler.

This is the main deployment script. It:
1. Initializes the VRAM allocator
2. Deploys all models from the MODELS configuration
3. Shows final VRAM state

Usage:
    python scripts/vram-scheduler/1_deploy_models.py
"""
import ray
import sys
import os

# Add vram-scheduler directory to path (folder has hyphen, can't be imported as package)
vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vram_scheduler_dir)

# Import modules directly (since folder name has hyphen)
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
        "vram_gb": 1.0,
        "replicas": 10  # Multiple replicas can share GPUs (fractional GPU allocation)
    },
}

def main():
    ray_address = os.getenv("RAY_ADDRESS", "ray://10.0.1.53:10001")
    ray.init(address=ray_address, ignore_reinit_error=True)
    
    # Ensure allocator exists
    print("Initializing VRAM allocator...")
    get_vram_allocator()
    
    # Deploy all models
    print("Deploying models...")
    orchestrator = ModelOrchestrator.remote()
    ray.get(orchestrator.apply.remote(MODELS))
    
    # Check VRAM state
    print("\nChecking VRAM state...")
    allocator = ray.get_actor("vram_allocator", namespace="system")
    state = ray.get(allocator.get_all_nodes.remote())
    
    print("\nVRAM State:")
    for node_id, info in state.items():
        print(f"  Node {node_id[:8]}: {info['free']:.2f}GB free / "
              f"{info['total']:.2f}GB total, "
              f"{len(info['allocs'])} models loaded")

if __name__ == "__main__":
    main()

