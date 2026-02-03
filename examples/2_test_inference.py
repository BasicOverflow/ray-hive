"""Test inference features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from ray_hive.inference import inference, inference_batch

MODEL_ID = "qwen-max-concurrency"
prompt = "Write a short poem about beer"
amount = 10_000

_ = inference(prompt, model_id=MODEL_ID, max_tokens=10, temperature=0.0)

prompts = [f"{prompt} {i}" for i in range(amount)]
start = time.time()
results = inference_batch(prompts, model_id=MODEL_ID, max_tokens=100, temperature=0.0)
elapsed = time.time() - start
print(f"Processed {len(results)} prompts in {elapsed:.3f}s ({len(results)/elapsed:.2f} req/s)")