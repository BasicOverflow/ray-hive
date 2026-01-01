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

# Add repo root to path so vram_scheduler can be imported
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

from scripts.vram_scheduler.model_orchestrator import ModelOrchestrator, MODELS
from scripts.vram_scheduler.vram_allocator import get_vram_allocator

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

