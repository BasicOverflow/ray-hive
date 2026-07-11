"""
Test gpu_registry actor — verifies VRAM DaemonSet reporting is working.
"""
import ray
import sys

RAY_ADDRESS = "ray://10.0.1.53:10001"


def main():
    print(f"Connecting to Ray cluster at {RAY_ADDRESS}...")
    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)

    try:
        try:
            registry = ray.get_actor("gpu_registry", namespace="system")
            print("✅ GPU registry actor found")
        except ValueError:
            print("❌ GPU registry actor not found. Is the DaemonSet running?")
            sys.exit(1)

        state = ray.get(registry.get_all_gpus.remote())

        if not state:
            print("⚠️  No VRAM data available yet. DaemonSet may still be initializing.")
            sys.exit(0)

        gpu_nodes = {k: v for k, v in state.items() if v and len(k) < 50 and not k.startswith('c')}

        print(f"VRAM State ({len(gpu_nodes)} GPUs):")
        print("-" * 60)
        for gpu_key, info in sorted(gpu_nodes.items()):
            print(f"GPU: {gpu_key}")
            print(f"  Total VRAM: {info.get('total', 0):.2f} GB")
            print(f"  Free VRAM:  {info.get('free', 0):.2f} GB")
            print(f"  Available:  {info.get('available', 0):.2f} GB")
            print(f"  Pending:    {sum(info.get('pending', {}).values()):.2f} GB")
            print(f"  Active:     {sum(info.get('active', {}).values()):.2f} GB")
            if info.get("specs", {}).get("name"):
                print(f"  Model:      {info['specs']['name']}")
            print()

        test_required = 5.0
        found = [k for k, v in gpu_nodes.items() if v.get("available", 0) >= test_required]
        if found:
            print(f"✅ Found GPU with {test_required}GB+ available VRAM: {found[0]}")
        else:
            print(f"⚠️  No GPU has {test_required}GB+ available VRAM")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
