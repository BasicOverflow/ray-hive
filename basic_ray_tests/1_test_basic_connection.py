"""
Basic Ray cluster connection test.

Verifies:
- Can connect to Ray cluster
- Cluster resources are visible
- Simple tasks can execute
"""
import ray
import sys

RAY_ADDRESS = "ray://10.0.1.53:10001"

def main():
    print(f"Connecting to Ray cluster at {RAY_ADDRESS}...")
    
    try:
        ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
        
        cluster = ray.cluster_resources()
        print(f"\nCluster Resources:")
        print(f"  CPUs: {cluster.get('CPU', 0)}")
        print(f"  Memory: {cluster.get('memory', 0) / (1024**3):.2f} GB")
        print(f"  GPUs: {cluster.get('GPU', 0)}")
        print(f"  Nodes: {len([k for k in cluster.keys() if k.startswith('node:')])}")
        
        available = ray.available_resources()
        print(f"\nAvailable Resources:")
        print(f"  CPUs: {available.get('CPU', 0)}")
        print(f"  Memory: {available.get('memory', 0) / (1024**3):.2f} GB")
        print(f"  GPUs: {available.get('GPU', 0)}")
        
        @ray.remote
        def hello():
            return "Hello from Ray!"
        
        print(f"\n✅ Test task: {ray.get(hello.remote(), timeout=30)}")
        print("✅ Basic connection test passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
