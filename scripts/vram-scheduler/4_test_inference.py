"""Test inference through the unified router endpoint with 100 concurrent requests."""
import os
import sys
import ray
from ray import serve
import time
import asyncio

vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vram_scheduler_dir)

import ray_utils

SUPPRESS_LOGGING = False
MODEL_ID = "max-llm"
NUM_REQUESTS = 100
MAX_RESPONSE_TOKENS = 100

def main():
    ray_utils.init_ray(suppress_logging=SUPPRESS_LOGGING)
    
    status = serve.status()
    deployment_name = None
    app_name = None
    
    if MODEL_ID in status.applications and status.applications[MODEL_ID].deployments:
        deployment_name = list(status.applications[MODEL_ID].deployments.keys())[0]
        app_name = MODEL_ID
    else:
        for app_name_check, app_info in status.applications.items():
            if app_name_check.startswith(f"{MODEL_ID}-gpu-") and app_info.deployments:
                deployment_name = list(app_info.deployments.keys())[0]
                app_name = app_name_check
                break
    
    if not deployment_name:
        raise RuntimeError(f"Could not find deployment for {MODEL_ID}. Available: {list(status.applications.keys())}")
    
    handle = serve.get_deployment_handle(deployment_name, app_name=app_name)
    test_prompt = "Write a long poem about beer"
    request = {"prompt": test_prompt, "max_tokens": MAX_RESPONSE_TOKENS, "temperature": 0.7}
    
    async def run_requests():
        responses = [handle.remote(request) for _ in range(NUM_REQUESTS)]
        return [await resp for resp in responses]
    
    print(f"Testing {MODEL_ID}: {NUM_REQUESTS} requests...")
    start_time = time.time()
    results = asyncio.run(run_requests())
    elapsed = time.time() - start_time
    
    successes = sum(1 for r in results if not (isinstance(r, dict) and "error" in r))
    failures = NUM_REQUESTS - successes
    
    print(f"Results: {successes}/{NUM_REQUESTS} success, {elapsed:.2f}s total, {NUM_REQUESTS/elapsed:.2f} req/s")
    
    # Try to get router stats if using router
    if app_name == MODEL_ID:
        try:
            router_handle = serve.get_deployment_handle(deployment_name, app_name=app_name)
            # Router has get_stats method, but we can't easily call it from here
            # The stats are tracked internally
            print("Note: Router tracks request distribution internally")
        except:
            pass
    
    output_file = f"inference_results_{int(time.time())}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Model: {MODEL_ID}\nPrompt: {test_prompt}\n")
        f.write(f"Requests: {NUM_REQUESTS}, Time: {elapsed:.2f}s, Throughput: {NUM_REQUESTS/elapsed:.2f} req/s\n")
        f.write(f"Successes: {successes}, Failures: {failures}\n\n")
        for i, result in enumerate(results, 1):
            f.write(f"\n--- Response #{i} ---\n")
            if isinstance(result, dict) and "error" in result:
                f.write(f"Error: {result['error']}\n")
            elif isinstance(result, dict) and "text" in result:
                f.write(f"{result['text']}\n")
            else:
                f.write(f"{result}\n")
    
    print(f"Results saved to: {output_file}")
    if isinstance(results[0], dict) and "text" in results[0]:
        print(f"Sample: {results[0]['text'][:200]}...")

if __name__ == "__main__":
    main()

