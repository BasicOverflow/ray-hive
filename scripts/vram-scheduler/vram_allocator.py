"""
Global VRAM Allocator Actor - singleton, HA-safe, persistent.

This actor maintains VRAM state across all GPU nodes and handles
reservation/release of VRAM for models.
"""
import ray
from typing import Dict, Optional

@ray.remote(num_cpus=0)
class VRAMAllocator:
    """Global VRAM allocator - singleton, HA-safe, persistent."""
    
    def __init__(self):
        # node_id -> {total_gb, free_gb, allocations: {model_id: gb}}
        self.nodes: Dict[str, Dict] = {}
    
    def update_node(self, node_id: str, free_gb: float, total_gb: float):
        """Update VRAM state for a node (called by DaemonSet)."""
        if node_id not in self.nodes:
            self.nodes[node_id] = {"total": total_gb, "free": free_gb, "allocs": {}}
        else:
            # Update free/total, but preserve allocations
            self.nodes[node_id]["total"] = total_gb
            # Free = actual_free - reserved
            reserved = sum(self.nodes[node_id]["allocs"].values())
            self.nodes[node_id]["free"] = free_gb - reserved
    
    def reserve(self, model_id: str, node_id: str, required_gb: float) -> bool:
        """Reserve VRAM for a model. Returns True if successful."""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        available = node["free"]
        
        if available < required_gb:
            return False
        
        # Reserve it
        node["free"] -= required_gb
        node["allocs"][model_id] = required_gb
        return True
    
    def release(self, model_id: str, node_id: str):
        """Release VRAM reservation for a model."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        if model_id in node["allocs"]:
            gb = node["allocs"].pop(model_id)
            node["free"] += gb
    
    def get_node_vram(self, node_id: str) -> Optional[Dict]:
        """Get VRAM info for a specific node."""
        return self.nodes.get(node_id)
    
    def find_node_with_vram(self, required_gb: float) -> Optional[str]:
        """Find a node with enough free VRAM (only K8s node names, not old Ray node IDs)."""
        for node_id, node in self.nodes.items():
            # Skip old Ray node IDs (long hex strings starting with 'c')
            if len(node_id) > 50 or node_id.startswith('c'):
                continue
            if node["free"] >= required_gb:
                return node_id
        return None
    
    def cleanup_old_ray_node_ids(self) -> int:
        """Remove old Ray node ID entries (long hex strings). Returns count removed."""
        old_ids = [node_id for node_id in self.nodes.keys() 
                   if len(node_id) > 50 or node_id.startswith('c')]
        for node_id in old_ids:
            del self.nodes[node_id]
        return len(old_ids)
    
    def get_all_nodes(self) -> Dict:
        """Get VRAM state for all nodes."""
        return self.nodes.copy()


def get_vram_allocator():
    """Get or create the global VRAM allocator actor."""
    try:
        return ray.get_actor("vram_allocator", namespace="system")
    except ValueError:
        # Create it if it doesn't exist
        return VRAMAllocator.options(
            name="vram_allocator",
            namespace="system",
            lifetime="detached"
        ).remote()

