"""Test inference on deployed models and load balancing."""
import ray
import os
import sys
import time
import warnings

# Suppress Ray Serve internal warnings about network latency
warnings.filterwarnings("ignore", category=UserWarning, module="ray.serve")

vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vram_scheduler_dir)

from ray import serve

RAY_ADDRESS = os.getenv("RAY_ADDRESS", "ray://10.0.1.53:10001")
PROMPT = "What is artificial intelligence?"
TIMEOUT = 300  # 5 minute timeout for inference

ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)

# Get deployed models
serve_status = serve.status()
apps = serve_status.applications

print(f"Found {len(apps)} model(s): {list[str](apps.keys())}\n")

# Test each model
for model_id in apps.keys():
    print(f"Testing {model_id}...")
    try:
        handle = serve.get_deployment_handle(model_id, app_name=model_id)
        response = handle.generate.remote(PROMPT)
        result = response.result()
        # Extract text from list if needed
        if isinstance(result, list) and len(result) > 0:
            text = result[0] if isinstance(result[0], str) else result[0]
        else:
            text = result
        print(f"Response: {text}\n")
    except Exception as e:
        print(f"Error testing {model_id}: {e}\n")





# Load balancing test
if apps:
    test_model = list(apps.keys())[0]
    print(f"Load balancing test: sending 50 parallel requests to {test_model}")
    
    try:
        handle = serve.get_deployment_handle(test_model, app_name=test_model)
        start = time.time()
        
        # Send parallel requests
        responses = [handle.generate.remote(f"{PROMPT} Request {i}") for i in range(50)]
        results = [r.result() for r in responses]
        
        elapsed = time.time() - start
        print(f"Completed 50 requests in {elapsed:.2f}s ({50/elapsed:.1f} req/s)")
        print(f"Received {len(results)} responses")
    except Exception as e:
        print(f"Error in load balancing test: {e}")

