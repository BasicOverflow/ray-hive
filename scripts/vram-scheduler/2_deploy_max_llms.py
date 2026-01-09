"""Deploy maximum replicas of a single model to fill available VRAM."""
import os
import sys
import ray

vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vram_scheduler_dir)
   


import ray_utils
import model_orchestrator
import vram_allocator

ModelOrchestrator = model_orchestrator.ModelOrchestrator
get_vram_allocator = vram_allocator.get_vram_allocator

SUPPRESS_LOGGING = False

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
ACTUAL_VRAM_GB = 3.89  # Total VRAM per replica (model + KV cache + overhead)
VRAM_BUFFER_GB = 0.1  # Hard buffer to leave free on every GPU (GB)

def main():
    ray_utils.init_ray(suppress_logging=SUPPRESS_LOGGING)
    
    # Calculate max replicas based on available VRAM per node
    allocator = get_vram_allocator()
    
    # Wait a moment for daemonset to update if needed
    import time
    time.sleep(1)
    
    state = ray.get(allocator.get_all_gpus.remote())
    
    # Calculate replicas per GPU, then sum
    replicas_per_gpu = []
    total_free_vram = 0
    total_reserved = 0
    
    print("\nVRAM State Details (Per-GPU):")
    for gpu_key, info in state.items():
        # Skip old Ray node IDs
        if len(gpu_key) > 50 or gpu_key.startswith('c'):
            continue
        
        gpu_total = info.get("total", 0)
        gpu_free = info.get("free", 0)  # Actual free from nvidia-smi
        gpu_available = info.get("available", 0)  # Free minus pending reservations
        gpu_pending = info.get("pending", 0)
        gpu_active = info.get("active", 0)
        total_free_vram += gpu_free
        total_reserved += gpu_pending + gpu_active
        
        print(f"  {gpu_key}:")
        print(f"    Total: {gpu_total:.2f}GB, Free: {gpu_free:.2f}GB, Available: {gpu_available:.2f}GB")
        print(f"    Pending: {gpu_pending:.2f}GB, Active: {gpu_active:.2f}GB")
        
        # Use available VRAM (free - pending) for calculation, minus hard buffer
        buffer_gb = VRAM_BUFFER_GB
        available_with_buffer = max(0, gpu_available - buffer_gb)
        if available_with_buffer >= ACTUAL_VRAM_GB:
            # Calculate how many replicas can fit based on available memory (with buffer)
            gpu_replicas = int(available_with_buffer / ACTUAL_VRAM_GB)
            replicas_per_gpu.append((gpu_key, gpu_replicas, gpu_available, buffer_gb, gpu_total))
    
    max_replicas = sum(replicas for _, replicas, _, _, _ in replicas_per_gpu)
    
    print(f"\nSummary:")
    print(f"  Total free VRAM: {total_free_vram:.2f} GB")
    print(f"  Total reserved VRAM: {total_reserved:.2f} GB")
    print(f"  VRAM per replica: {ACTUAL_VRAM_GB:.2f} GB")
    print(f"  VRAM buffer: {VRAM_BUFFER_GB:.2f}GB per GPU")
    print(f"  Replicas per GPU:")
    for gpu_key, gpu_replicas, gpu_available, buffer_gb, gpu_total in replicas_per_gpu:
        available_with_buffer = max(0, gpu_available - buffer_gb)
        total_vram_used = gpu_replicas * ACTUAL_VRAM_GB
        print(f"    {gpu_key}: {gpu_replicas} replicas - {total_vram_used:.2f}GB used from {available_with_buffer:.2f}GB (included {VRAM_BUFFER_GB:.2f}GB buffer deducted from {gpu_available:.2f}GB available)")
    print(f"  Can deploy {max_replicas} replicas total")
    
    # input("hold")

    if max_replicas > 0:
        # Create model config - no pre-assignments, let replicas find GPUs dynamically
        # Ray Serve doesn't guarantee placement anyway, so pre-calculating doesn't help
        MODELS = {
            "max-llm": {
                "name": MODEL_NAME,
                "vram_gb": ACTUAL_VRAM_GB,  # Pass actual VRAM directly, no multipliers
                "replicas": max_replicas
            }
        }
        
        # Deploy using ModelOrchestrator (same approach as 1_deploy_models.py)
        orchestrator = ModelOrchestrator.remote()
        ray.get(orchestrator.apply.remote(MODELS))
        
        print(f"✅ Deployed with {max_replicas} replicas")
    else:
        print("❌ Not enough VRAM to deploy any replicas")

if __name__ == "__main__":
    main()

