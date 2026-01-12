"""Test inference features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import asyncio
from pydantic import BaseModel
from vram_scheduler.inference import inference, a_inference, inference_batch, a_inference_batch
from vram_scheduler.utils.ray_utils import init_ray

# Connect to Ray cluster
import ray
if not ray.is_initialized():
    init_ray(address=os.getenv("RAY_ADDRESS", "ray://10.0.1.53:10001"), suppress_logging=False)

MODEL_ID = "qwen-custom"
prompt = "Write a short poem about beer"

# # Synchronous inference
# result = inference(prompt, model_id=MODEL_ID, max_tokens=100, temperature=0.7)
# print(result[:200])

# # Async inference
# async def test_async():
#     return await a_inference(prompt, model_id=MODEL_ID, max_tokens=100, temperature=0.8)
# result_async = asyncio.run(test_async())
# print(result_async[:200])

# Batch inference
import time

prompts = [f"{prompt} {i}" for i in range(10_000)]

# Warmup: Do a small inference first to initialize connections and avoid measuring setup time
print("Warming up (initializing connections)...")
warmup_prompts = prompts[:1]
_ = inference_batch(warmup_prompts, model_id=MODEL_ID, max_tokens=10, temperature=0.0)

# Now time only the actual batch inference (connections are already established)
print("Starting timed batch inference...")
start_time = time.time()
results_batch = inference_batch(
    prompts, 
    model_id=MODEL_ID, 
    max_tokens=100, 
    temperature=0.0
    # batch_size=96  # Optional: match your max_num_seqs setting for optimal batching
)
end_time = time.time()
inference_time = end_time - start_time
requests_per_sec = len(results_batch) / inference_time if inference_time > 0 else float('inf')

# Write timing results at the top of the output file
output_file = "cluster_max_1000_poems_batch_inference_results.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Processed {len(results_batch)} prompts\n")
    f.write(f"Inference time (excluding setup): {inference_time:.3f} seconds\n")
    f.write(f"Requests processed per second: {requests_per_sec:.2f}\n")
    f.write("=" * 40 + "\n")
    for idx, result in enumerate(results_batch):
        f.write(f"Prompt {idx}: {prompts[idx]}\n")
        f.write(f"Result: {result}\n")
        f.write("-" * 40 + "\n")
print(f"Processed {len(results_batch)} prompts and wrote results to {output_file}")
print(f"Inference time (excluding setup): {inference_time:.3f} seconds")
print(f"Requests processed per second: {requests_per_sec:.2f}")

# # Async batch inference
# async def test_batch():
#     prompts = [prompt] * 50
#     return await a_inference_batch(
#         prompts,
#         model_id=MODEL_ID,
#         max_tokens=100,
#         temperature=0.7,
#         stop=["\n\n"]
#     )
# results = asyncio.run(test_batch())
# print(f"Processed {len(results)} async requests")

# # Structured output
# class MathResponse(BaseModel):
#     answer: str
#     explanation: str

# try:
#     structured = inference(
#         "What is 2+2? Explain.",
#         model_id=MODEL_ID,
#         structured_output=MathResponse,
#         max_tokens=150
#     )
#     print(f"Answer: {structured.answer}")
# except Exception as e:
#     print(f"Structured output failed: {e}")
