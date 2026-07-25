"""
Test gpu_registry actor — verifies VRAM DaemonSet reporting is working.
"""
import os
import ray
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

RAY_ADDRESS = os.environ["RAY_ADDRESS"]


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

        print(f"VRAM State ({len(state)} GPUs):")
        print("-" * 60)
        for gpu_key, info in sorted(state.items()):
            print(f"GPU: {gpu_key}")
            print(f"  Total VRAM: {info.get('total', 0):.2f} GB")
            print(f"  Free VRAM:  {info.get('free', 0):.2f} GB")
            print(f"  Available:  {info.get('available', 0):.2f} GB")
            print(f"  Pending:    {sum(info.get('pending', {}).values()):.2f} GB")
            print(f"  Active:     {sum(info.get('active', {}).values()):.2f} GB")
            # print("  PyCUDA specs:")
            # for name, value in sorted(info.get("specs", {}).items()):
            #     print(f"    {name}: {value}")
            # print()

        test_required = 5.0
        found = [k for k, v in state.items() if v.get("available", 0) >= test_required]
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
