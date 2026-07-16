"""Test full deployment with multiple replicas and 'max' replicas."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.inference import inference_batch

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

scheduler = RayHive(suppress_logging=True)

MODEL_ID = "qwen-full-deployment"
model_name = "Qwen/Qwen3-0.6B-GPTQ-Int8"

prompt = "Write a short poem about beer"
amount = 50_000
prompts = [f"{prompt} {i}" for i in range(amount)]


print("\n" + "=" * 80)
print("Test: Deploy with replicas=-1 (all available GPUs)")
print("=" * 80)

scheduler.deploy_model(
    model_id=MODEL_ID,
    model_name=model_name,
    max_input_prompt_length=1024,
    max_output_prompt_length=2048,
    replicas=-1,
    swap_space_per_instance=10,
)


_ = inference_batch(prompts[:10], model_id=MODEL_ID)
time.sleep(2)

start = time.time()
results = inference_batch(prompts, model_id=MODEL_ID, temperature=0.7, top_k=50, top_p=0.9)
elapsed = time.time() - start

throughput = len(results) / elapsed
print(f"\nDeployment Throughput: {throughput:.2f} req/s")
print(f"Processed {len(results)} prompts in {elapsed:.3f}s")

scheduler.shutdown(MODEL_ID)

print("\n" + "=" * 80)
print("Full deployment test completed")
print("=" * 80)

