"""Shutdown functionality for VRAM scheduler."""
import ray
from ray import serve
import time
from typing import Optional
from .core.vram_allocator import get_vram_allocator


def kill_vram_allocator():
    """Kill VRAM allocator actor so it gets recreated fresh."""
    try:
        allocator = ray.get_actor("vram_allocator", namespace="system")
        ray.kill(allocator)
        time.sleep(0.5)
        print("✅ Killed VRAM allocator actor (will be recreated fresh)")
    except ValueError:
        pass


def shutdown_all():
    """Shutdown all Ray Serve applications."""
    allocator = get_vram_allocator()
    apps = serve.status().applications
    
    if apps:
        print(f"Shutting down {len(apps)} application(s): {list(apps.keys())}")
        serve.shutdown()
        print("✅ All applications shut down")
    
    cleared = ray.get(allocator.clear_all_reservations.remote())
    if cleared > 0:
        print(f"✅ Cleared {cleared} VRAM reservations")


def shutdown_model(model_id: str):
    """Shutdown a specific model deployment."""
    allocator = get_vram_allocator()
    apps = serve.status().applications
    
    if model_id not in apps:
        print(f"⚠️  Model '{model_id}' not found in deployments")
        return
    
    serve.delete(name=model_id)
    print(f"✅ Shut down {model_id}")
    
    cleared = ray.get(allocator.clear_reservations_by_prefix.remote(f"{model_id}-"))
    if cleared > 0:
        print(f"✅ Cleared {cleared} VRAM reservations for {model_id}")

