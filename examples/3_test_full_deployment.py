"""Test full deployment with multiple replicas and 'max' replicas."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ray_hive import RayHive
from ray_hive.inference import inference_batch

scheduler = RayHive(suppress_logging=True)

MODEL_ID = "qwen-full-deployment"
model_name = "Qwen/Qwen3-0.6B-GPTQ-Int8"
vram_weights_gb = 0.763

prompt = "Write a short poem about beer"
amount = 300_000
prompts = [f"{prompt} {i}" for i in range(amount)]


print("\n" + "=" * 80)
print("Test 2: Deploy with 'max' replicas (all available GPUs)")
print("=" * 80)

scheduler.deploy_model(
    model_id=MODEL_ID,
    model_name=model_name,
    vram_weights_gb=vram_weights_gb,
    max_input_prompt_length=1024,
    max_output_prompt_length=2048,
    max_num_seqs=850,
    max_num_batched_tokens=16384,
    replicas="max",
    swap_space_per_instance=10,
)


_ = inference_batch(prompts[:10], model_id=MODEL_ID, max_tokens=50, temperature=0.0)
time.sleep(2)

start = time.time()
results = inference_batch(prompts, model_id=MODEL_ID, max_tokens=50, temperature=0.0)
elapsed = time.time() - start

throughput = len(results) / elapsed
print(f"\nDeployment Throughput: {throughput:.2f} req/s")
print(f"Processed {len(results)} prompts in {elapsed:.3f}s")

scheduler.shutdown(MODEL_ID)

print("\n" + "=" * 80)
print("Full deployment test completed")
print("=" * 80)

