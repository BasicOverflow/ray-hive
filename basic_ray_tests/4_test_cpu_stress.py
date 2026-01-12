"""
CPU stress test - verifies task distribution across cluster.

Tests:
- Parallel CPU task execution
- Task distribution across CPU workers
- CPU worker performance
"""
import ray
import time
import sys
from collections import Counter

RAY_ADDRESS = "ray://10.0.1.53:10001"

@ray.remote
def cpu_task(n):
    """CPU-intensive task that also reports which node it ran on."""
    result = 1
    for i in range(1, n + 1):
        result *= i
    
    # Get the node ID where this task ran
    node_id = ray.get_runtime_context().get_node_id()
    return {"result": result, "node_id": node_id}

def main():
    print(f"Connecting to Ray cluster at {RAY_ADDRESS}...")
    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
    
    try:
        num_tasks = 20
        print(f"Running {num_tasks} CPU-intensive tasks...")
        
        start = time.time()
        results = ray.get([cpu_task.remote(50000) for _ in range(num_tasks)], timeout=300)
        elapsed = time.time() - start
        
        # Count tasks per node
        node_counts = Counter(r["node_id"] for r in results)
        
        print(f"\n✅ Completed {num_tasks} tasks in {elapsed:.2f}s")
        print(f"\nTask distribution across nodes:")
        for node_id, count in sorted(node_counts.items()):
            print(f"  Node {node_id[:8]}: {count} tasks ({count/num_tasks*100:.1f}%)")
        
        if len(node_counts) > 1:
            print(f"\n✅ Tasks were distributed across {len(node_counts)} nodes")
        else:
            print(f"\n⚠️  All tasks ran on a single node (may indicate scheduling issue)")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
