"""Global VRAM Allocator Actor - tracks VRAM state per GPU."""
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
                "active": {}
            }
        else:
            self.gpus[gpu_key]["total"] = total_gb
            self.gpus[gpu_key]["free"] = free_gb
    
    def get_available_vram(self, gpu_key: str) -> float:
        """Get available VRAM: nvidia-smi free minus pending reservations."""
        if gpu_key not in self.gpus:
            return 0.0
        gpu = self.gpus[gpu_key]
        pending_total = sum(gpu["pending"].values())
        return max(0.0, gpu["free"] - pending_total)
    
    def reserve(self, replica_id: str, gpu_key: str, required_gb: float) -> bool:
        """Reserve VRAM for a replica. Returns True if successful."""
        if gpu_key not in self.gpus:
            return False
        
        gpu = self.gpus[gpu_key]
        available = self.get_available_vram(gpu_key)
        
        if available < required_gb:
            return False
        
        gpu["pending"][replica_id] = required_gb
        return True
    
    def mark_initialized(self, replica_id: str, gpu_key: str):
        """Mark replica as initialized. Moves reservation from pending to active."""
        if gpu_key not in self.gpus:
            return
        
        gpu = self.gpus[gpu_key]
        if replica_id in gpu["pending"]:
            required_gb = gpu["pending"].pop(replica_id)
            gpu["active"][replica_id] = required_gb
    
    def release(self, replica_id: str, gpu_key: str):
        """Release VRAM reservation for a replica."""
        if gpu_key not in self.gpus:
            return
        
        gpu = self.gpus[gpu_key]
        if replica_id in gpu["pending"]:
            gpu["pending"].pop(replica_id)
        if replica_id in gpu["active"]:
            gpu["active"].pop(replica_id)
    
    def get_gpu_vram(self, gpu_key: str) -> Optional[Dict]:
        """Get VRAM info for a GPU."""
        if gpu_key not in self.gpus:
            return None
        
        gpu = self.gpus[gpu_key]
        available = self.get_available_vram(gpu_key)
        return {
            "total": gpu["total"],
            "free": gpu["free"],  # Actual free from nvidia-smi
            "available": available,  # Free minus pending reservations
            "pending": sum(gpu["pending"].values()),
            "active": sum(gpu["active"].values()),
            "pending_count": len(gpu["pending"]),
            "active_count": len(gpu["active"])
        }
    
    def find_gpu_with_vram(self, required_gb: float, node_id: Optional[str] = None) -> Optional[str]:
        """Find GPU with enough VRAM. If node_id provided, only searches that node."""
        candidates = []
        for gpu_key, gpu in self.gpus.items():
            # Skip old Ray node IDs
            if len(gpu_key) > 50 or gpu_key.startswith('c'):
                continue
            
            if node_id and not gpu_key.startswith(node_id + ":"):
                continue
            
            available = self.get_available_vram(gpu_key)
            if available >= required_gb:
                candidates.append((gpu_key, available))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def get_all_gpus(self) -> Dict:
        """Get VRAM state for all GPUs."""
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
                "active_count": len(gpu["active"])
            }
        return result
    
    def clear_all_reservations(self) -> int:
        """Clear all reservations. Returns number cleared."""
        total_cleared = 0
        for gpu_key in self.gpus:
            gpu = self.gpus[gpu_key]
            total_cleared += len(gpu["pending"]) + len(gpu["active"])
            gpu["pending"] = {}
            gpu["active"] = {}
        self.gpu_locks.clear()
        return total_cleared
    
    def clear_reservations_by_prefix(self, prefix: str) -> int:
        """Clear reservations starting with prefix."""
        cleared = 0
        for gpu_key in self.gpus:
            gpu = self.gpus[gpu_key]
            to_remove = [rid for rid in gpu["pending"].keys() if rid.startswith(prefix)]
            for rid in to_remove:
                gpu["pending"].pop(rid)
                cleared += 1
            to_remove = [rid for rid in gpu["active"].keys() if rid.startswith(prefix)]
            for rid in to_remove:
                gpu["active"].pop(rid)
                cleared += 1
            if gpu_key in self.gpu_locks:
                lock_holder = self.gpu_locks[gpu_key]
                if lock_holder and lock_holder.startswith(prefix):
                    self.gpu_locks[gpu_key] = None
        return cleared
    
    def acquire_gpu_lock(self, gpu_key: str, replica_id: str) -> bool:
        """Acquire initialization lock for a GPU. Returns True if acquired."""
        if gpu_key not in self.gpu_locks:
            self.gpu_locks[gpu_key] = replica_id
            return True
        
        current_lock_holder = self.gpu_locks[gpu_key]
        
        if current_lock_holder is None or current_lock_holder == replica_id:
            self.gpu_locks[gpu_key] = replica_id
            return True
        
        if gpu_key not in self.gpus:
            self.gpu_locks[gpu_key] = replica_id
            return True
        
        gpu = self.gpus[gpu_key]
        if current_lock_holder not in gpu["active"]:
            self.gpu_locks[gpu_key] = replica_id
            return True
        
        return False
    
    def release_gpu_lock(self, gpu_key: str, replica_id: str):
        """Release initialization lock for a GPU."""
        if gpu_key in self.gpu_locks and self.gpu_locks[gpu_key] == replica_id:
            self.gpu_locks[gpu_key] = None
    
    def clear_gpu_locks(self, gpu_key: str = None):
        """Clear GPU lock(s). If gpu_key is None, clears all locks."""
        if gpu_key is None:
            self.gpu_locks.clear()
        elif gpu_key in self.gpu_locks:
            self.gpu_locks[gpu_key] = None
    


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

