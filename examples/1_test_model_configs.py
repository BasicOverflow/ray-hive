"""Deploy/inference configs — single GPU, all GPUs, and 2 replicas."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.inference import inference_batch

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

scheduler = RayHive(suppress_logging=True)

deployments = [
    # {
    #     "model_id": "qwen-single-gpu",
    #     "description": "Single GPU",
    #     "config": {
    #         # "gpu": "ergos-06-nv:gpu0", # pin to specific GPU
    #         "model_name": "Qwen/Qwen3-0.6B-GPTQ-Int8",
    #         "max_input_prompt_length": 1024,
    #         "max_output_prompt_length": 2048,
    #         "replicas": 1,
    #     },
    # },
    {
        "model_id": "qwen-two-replicas",
        "description": "replicas=2",
        "config": {
            "model_name": "Qwen/Qwen3-0.6B-GPTQ-Int8",
            "max_input_prompt_length": 1024,
            "max_output_prompt_length": 2048,
            # "replicas": 2,
            "gpu":["ergos-06-nv:gpu0", "ergos-02-nv:gpu0"]
        },
    },
    # {
    #     "model_id": "qwen-all-gpus",
    #     "description": "replicas=-1 (all eligible GPUs)",
    #     "config": {
    #         "model_name": "Qwen/Qwen3-0.6B-GPTQ-Int8",
    #         "max_input_prompt_length": 1024,
    #         "max_output_prompt_length": 2048,
    #         "replicas": -1,
    #     },
    # },
]

prompt = "Write a short poem about beer"
amount = 1_000
prompts = [f"{prompt} {i}" for i in range(amount)]

for idx, deployment in enumerate(deployments):
    model_id = deployment["model_id"]
    print(f"\n=== {deployment['description']} ({model_id}) ===")

    scheduler.deploy_model(model_id=model_id, **deployment["config"])
    time.sleep(2)

    _ = inference_batch(prompts, model_id=model_id, max_tokens=100, temperature=0.0)
    time.sleep(2)

    start = time.time()
    results = inference_batch(prompts, model_id=model_id, max_tokens=100, temperature=0.0)
    elapsed = time.time() - start
    print(f"Processed {len(results)} prompts in {elapsed:.3f}s ({len(results)/elapsed:.2f} req/s)")

    scheduler.shutdown(model_id)
    if idx < len(deployments) - 1:
        time.sleep(3)
