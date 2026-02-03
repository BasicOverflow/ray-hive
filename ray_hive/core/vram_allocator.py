# Global VRAM Allocator Actor - tracks VRAM state per GPU
# Singleton actor (detached, HA-safe) that maintains VRAM state across all nodes
# Tracks per-GPU VRAM (free, total, pending, active) and GPU hardware specs

import ray
from typing import Dict, Optional

@ray.remote(num_cpus=0)
class VRAMAllocator:
    """Global VRAM allocator - singleton, HA-safe, persistent."""
    
    def __init__(self):
        self.gpus: Dict[str, Dict] = {}
        self.gpu_locks: Dict[str, Optional[str]] = {}
    
    def update_gpu(self, node_id: str, gpu_id: int, free_gb: float, total_gb: float):
        """Update VRAM state for a GPU (called by DaemonSet)."""
        gpu_key = f"{node_id}:gpu{gpu_id}"
        if gpu_key not in self.gpus:
            self.gpus[gpu_key] = {
                "total": total_gb, 
                "free": free_gb,
                "pending": {},
                "active": {},
                "sm_count": None,
                "gpu_model": None,
                "compute_cap": None,
                "l2_cache_kb": None
            }
        else:
            self.gpus[gpu_key]["total"] = total_gb
            self.gpus[gpu_key]["free"] = free_gb
    
    def update_gpu_specs(self, node_id: str, gpu_id: int, sm_count: Optional[int] = None, 
                         gpu_model: Optional[str] = None, compute_cap: Optional[str] = None,
                         l2_cache_kb: Optional[float] = None):
        """Update GPU hardware specs (called by DaemonSet)."""
        gpu_key = f"{node_id}:gpu{gpu_id}"
        if gpu_key not in self.gpus:
            self.gpus[gpu_key] = {
                "total": 0.0,
                "free": 0.0,
                "pending": {},
                "active": {},
                "sm_count": sm_count,
                "gpu_model": gpu_model,
                "compute_cap": compute_cap,
                "l2_cache_kb": l2_cache_kb
            }
        else:
            if sm_count is not None:
                self.gpus[gpu_key]["sm_count"] = sm_count
            if gpu_model is not None:
                self.gpus[gpu_key]["gpu_model"] = gpu_model
            if compute_cap is not None:
                self.gpus[gpu_key]["compute_cap"] = compute_cap
            if l2_cache_kb is not None:
                self.gpus[gpu_key]["l2_cache_kb"] = l2_cache_kb
    
    def get_available_vram(self, gpu_key: str) -> float:
        if gpu_key not in self.gpus:
            return 0.0
        gpu = self.gpus[gpu_key]
        return gpu["free"] - sum(gpu["pending"].values())
    
    def reserve(self, replica_id: str, gpu_key: str, required_gb: float) -> bool:
        if gpu_key not in self.gpus:
            return False
        if self.get_available_vram(gpu_key) < required_gb:
            return False
        self.gpus[gpu_key]["pending"][replica_id] = required_gb
        return True
    
    def mark_initialized(self, replica_id: str, gpu_key: str):
        if gpu_key not in self.gpus:
            return
        gpu = self.gpus[gpu_key]
        if replica_id in gpu["pending"]:
            gpu["active"][replica_id] = gpu["pending"].pop(replica_id)
    
    def release(self, replica_id: str, gpu_key: str):
        if gpu_key not in self.gpus:
            return
        gpu = self.gpus[gpu_key]
        gpu["pending"].pop(replica_id, None)
        gpu["active"].pop(replica_id, None)
    
    def get_gpu_vram(self, gpu_key: str) -> Optional[Dict]:
        if gpu_key not in self.gpus:
            return None
        gpu = self.gpus[gpu_key]
        return {
            "total": gpu["total"],
            "free": gpu["free"],
            "available": self.get_available_vram(gpu_key),
            "pending": sum(gpu["pending"].values()),
            "active": sum(gpu["active"].values()),
            "pending_count": len(gpu["pending"]),
            "active_count": len(gpu["active"]),
            "sm_count": gpu.get("sm_count"),
            "gpu_model": gpu.get("gpu_model"),
            "compute_cap": gpu.get("compute_cap"),
            "l2_cache_kb": gpu.get("l2_cache_kb")
        }
    
    def find_gpu_with_vram(self, required_gb: float, node_id: Optional[str] = None) -> Optional[str]:
        candidates = []
        for gpu_key, gpu in self.gpus.items():
            if len(gpu_key) > 50 or gpu_key.startswith('c'):
                continue
            if node_id and not gpu_key.startswith(node_id + ":"):
                continue
            available = self.get_available_vram(gpu_key)
            if available >= required_gb:
                candidates.append((gpu_key, available))
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1])[0]
    
    def get_all_gpus(self) -> Dict:
        """Get VRAM state for all GPUs, including hardware specs."""
        result = {}
        for gpu_key, gpu in self.gpus.items():
            available = self.get_available_vram(gpu_key)
            result[gpu_key] = {
                "total": gpu["total"],
                "free": gpu["free"],
                "available": available,
                "pending": sum(gpu["pending"].values()),
                "active": sum(gpu["active"].values()),
                "pending_count": len(gpu["pending"]),
                "active_count": len(gpu["active"]),
                "sm_count": gpu.get("sm_count"),
                "gpu_model": gpu.get("gpu_model"),
                "compute_cap": gpu.get("compute_cap"),
                "l2_cache_kb": gpu.get("l2_cache_kb")
            }
        return result
    
    def clear_all_reservations(self) -> int:
        total_cleared = 0
        for gpu in self.gpus.values():
            total_cleared += len(gpu["pending"]) + len(gpu["active"])
            gpu["pending"] = {}
            gpu["active"] = {}
        self.gpu_locks.clear()
        return total_cleared
    
    def clear_reservations_by_prefix(self, prefix: str) -> int:
        cleared = 0
        for gpu_key, gpu in self.gpus.items():
            for rid in list(gpu["pending"].keys()):
                if rid.startswith(prefix):
                    gpu["pending"].pop(rid)
                    cleared += 1
            for rid in list(gpu["active"].keys()):
                if rid.startswith(prefix):
                    gpu["active"].pop(rid)
                    cleared += 1
            if gpu_key in self.gpu_locks and self.gpu_locks[gpu_key] and self.gpu_locks[gpu_key].startswith(prefix):
                self.gpu_locks[gpu_key] = None
        return cleared
    
    


def get_vram_allocator():
    """Get or create global VRAM allocator actor."""
    try:
        return ray.get_actor("vram_allocator", namespace="system")
    except ValueError:
        return VRAMAllocator.options(
            name="vram_allocator",
            namespace="system",
            lifetime="detached"
        ).remote()

