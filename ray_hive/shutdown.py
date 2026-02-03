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
    except ValueError:
        pass


def shutdown_all():
    """Shutdown all Ray Serve applications."""
    allocator = get_vram_allocator()
    apps = serve.status().applications
    
    if apps:
        serve.shutdown()
    
    ray.get(allocator.clear_all_reservations.remote())


def shutdown_model(model_id: str):
    """Shutdown a specific model deployment and all related apps."""
    allocator = get_vram_allocator()
    apps = serve.status().applications
    
    # Find all apps related to this model_id
    apps_to_delete = []
    for app_name in apps.keys():
        if app_name == model_id or app_name.startswith(f"{model_id}-"):
            apps_to_delete.append(app_name)
    
    if not apps_to_delete:
        return
    
    for app_name in apps_to_delete:
        try:
            serve.delete(name=app_name)
        except Exception:
            pass
    
    ray.get(allocator.clear_reservations_by_prefix.remote(f"{model_id}-"))

