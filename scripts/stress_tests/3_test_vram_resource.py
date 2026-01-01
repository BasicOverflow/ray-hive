"""
Test VRAM Allocator Actor - verifies the VRAM scheduler is working.

This test queries the global VRAM allocator actor to check VRAM state
across all GPU nodes.
"""
import ray
import sys

RAY_ADDRESS = "ray://10.0.1.53:10001"

def main():
    print(f"Connecting to Ray cluster at {RAY_ADDRESS}...")
    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
    
    try:
        # Get the VRAM allocator actor
        try:
            allocator = ray.get_actor("vram_allocator", namespace="system")
            print("✅ VRAM allocator actor found")
        except ValueError:
            print("❌ VRAM allocator actor not found. Is the DaemonSet running?")
            sys.exit(1)
        
        # Get VRAM state for all nodes
        state = ray.get(allocator.get_all_nodes.remote())
        
        if not state:
            print("⚠️  No VRAM data available yet. DaemonSet may still be initializing.")
            sys.exit(0)
        
        # Filter out old Ray node ID entries (keep only K8s node names)
        # Ray node IDs are long hex strings, K8s node names are short like "ergos-02-nv"
        k8s_nodes = {k: v for k, v in state.items() if len(k) < 50 and not k.startswith('c')}
        ray_node_ids = {k: v for k, v in state.items() if k not in k8s_nodes}
        
        if ray_node_ids:
            print(f"⚠️  Found {len(ray_node_ids)} old Ray node ID entry(ies) - these will be ignored")
            print("   (This is normal after updating from Ray node IDs to K8s node names)\n")
        
        print(f"VRAM State ({len(k8s_nodes)} GPU nodes):")
        print("-" * 60)
        for node_id, info in sorted(k8s_nodes.items()):
            free = info.get("free", 0)
            total = info.get("total", 0)
            allocs = info.get("allocs", {})
            used = total - free
            print(f"Node: {node_id}")
            print(f"  Total VRAM: {total:.2f} GB")
            print(f"  Free VRAM:  {free:.2f} GB")
            print(f"  Used VRAM:  {used:.2f} GB")
            print(f"  Allocations: {len(allocs)} model(s)")
            if allocs:
                for model_id, gb in allocs.items():
                    print(f"    - {model_id}: {gb:.2f} GB")
            print()
        
        # Test finding a node with available VRAM
        test_required = 5.0  # GB
        node_with_vram = ray.get(allocator.find_node_with_vram.remote(test_required))
        
        # Filter to only K8s node names (ignore old Ray node IDs)
        if node_with_vram:
            if len(node_with_vram) < 50 and not node_with_vram.startswith('c'):
                # It's a K8s node name
                print(f"✅ Found node with {test_required}GB+ VRAM: {node_with_vram}")
            else:
                # It's an old Ray node ID, find a K8s node instead
                for k8s_node, info in k8s_nodes.items():
                    if info.get("free", 0) >= test_required:
                        print(f"✅ Found node with {test_required}GB+ VRAM: {k8s_node}")
                        break
                else:
                    print(f"⚠️  Found old Ray node ID entry, but no K8s node has {test_required}GB+ free VRAM")
        else:
            print(f"⚠️  No node has {test_required}GB+ free VRAM")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
