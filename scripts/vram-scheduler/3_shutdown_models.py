"""
Shutdown Ray Serve deployments.

Shuts down all deployed models and releases VRAM reservations.

Usage:
    python scripts/vram-scheduler/3_shutdown_models.py
    # Or shutdown specific app:
    python scripts/vram-scheduler/3_shutdown_models.py --app tinyllama
"""
import ray
import os
import sys
import argparse

# Add vram-scheduler directory to path
vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vram_scheduler_dir)

from ray import serve

def main():
    parser = argparse.ArgumentParser(description="Shutdown Ray Serve deployments")
    parser.add_argument("--app", type=str, help="Specific app name to shutdown (default: all)")
    args = parser.parse_args()
    
    ray_address = os.getenv("RAY_ADDRESS", "ray://10.0.1.53:10001")
    ray.init(address=ray_address, ignore_reinit_error=True)
    
    try:
        serve_status = serve.status()
        apps = serve_status.applications
        
        if not apps:
            print("No Ray Serve applications to shutdown.")
            return
        
        if args.app:
            if args.app in apps:
                print(f"Shutting down application: {args.app}")
                try:
                    # Try deleting specific app
                    serve.delete(name=args.app)
                    print(f"✅ Shut down {args.app}")
                except AttributeError:
                    # Fallback: shutdown all if delete doesn't work
                    print("Note: serve.delete() not available, shutting down all applications...")
                    serve.shutdown()
                    print("✅ All applications shut down")
            else:
                print(f"❌ Application '{args.app}' not found.")
                print(f"Available applications: {list(apps.keys())}")
        else:
            print("Shutting down all Ray Serve applications...")
            print(f"Found {len(apps)} application(s): {list(apps.keys())}")
            serve.shutdown()
            print("✅ All applications shut down")
            
    except Exception as e:
        print(f"Error shutting down deployments: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

