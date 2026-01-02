"""
Test inference on deployed models and load balancing.

Usage:
    python scripts/vram-scheduler/4_test_inference.py
"""
import ray
import os
import sys
import time

# Add vram-scheduler directory to path
vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vram_scheduler_dir)

from ray import serve

RAY_ADDRESS = os.getenv("RAY_ADDRESS", "ray://10.0.1.53:10001")
PROMPT = "What is artificial intelligence?"

ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)

# Get deployed models
serve_status = serve.status()
apps = serve_status.applications

print(f"Found {len(apps)} model(s): {list(apps.keys())}\n")

# Test each model
for model_id in apps.keys():
    print(f"Testing {model_id}...")
    handle = serve.get_deployment(model_id).get_handle()
    result = ray.get(handle.generate.remote(PROMPT))
    print(f"Response: {result}\n")

# Pick first model for load balancing test
if apps:
    test_model = list(apps.keys())[0]
    print(f"Load balancing test: sending 50 parallel requests to {test_model}")
    
    handle = serve.get_deployment(test_model).get_handle()
    start = time.time()
    
    # Send parallel requests
    futures = [handle.generate.remote(f"{PROMPT} Request {i}") for i in range(50)]
    results = ray.get(futures)
    
    elapsed = time.time() - start
    print(f"Completed 50 requests in {elapsed:.2f}s ({50/elapsed:.1f} req/s)")
    print(f"Received {len(results)} responses")

