"""Shutdown Ray Serve deployments."""
import ray
import os
import argparse
from ray import serve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=str, help="Specific app name to shutdown (default: all)")
    args = parser.parse_args()
    
    ray.init(address=os.getenv("RAY_ADDRESS", "ray://10.0.1.53:10001"), ignore_reinit_error=True)
    
    apps = serve.status().applications
    if not apps:
        print("No Ray Serve applications to shutdown.")
        return
    
    if args.app:
        if args.app in apps:
            try:
                serve.delete(name=args.app)
                print(f"✅ Shut down {args.app}")
            except AttributeError:
                serve.shutdown()
                print("✅ All applications shut down")
        else:
            print(f"❌ Application '{args.app}' not found. Available: {list(apps.keys())}")
    else:
        print(f"Shutting down {len(apps)} application(s): {list(apps.keys())}")
        serve.shutdown()
        print("✅ All applications shut down")

if __name__ == "__main__":
    main()

