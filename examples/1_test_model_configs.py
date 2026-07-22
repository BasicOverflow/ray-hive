"""Deploy/inference configs — 2 pinned replicas, CPU RAM spill/KV, and all GPUs."""
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.core.ray_gpu_alloc import RayPerformanceAllocator
from ray_hive.inference import inference_batch

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

scheduler = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=True)

PIN_GPUS = ["ergos-06-nv:gpu0", "ergos-02-nv:gpu0"]

deployments = [
    # {
    #     "model_id": "qwen-two-replicas",
    #     "description": "replicas=2 (pinned)",
    #     "config": {
    #         "model_name": "Qwen/Qwen3-0.6B-FP8",
    #         "max_input_prompt_length": 1024,
    #         "max_output_prompt_length": 2048,
    #         "replicas": 2,
    #         "gpu": PIN_GPUS,
    #         "allocation_cls": RayPerformanceAllocator,
    #         "trust_remote_code": True,
    #         "reasoning_parser": "qwen3",
    #         "default_chat_template_kwargs": {"enable_thinking": False},
    #     },
    # },
    # {
    #     "model_id": "qwen-cpu-ram",
    #     "description": "replicas=2 + cpu_ram_per_instance=4",
    #     "config": {
    #         "model_name": "Qwen/Qwen3-0.6B-FP8",
    #         "max_input_prompt_length": 1024,
    #         "max_output_prompt_length": 2048,
    #         "replicas": 2,
    #         "gpu": PIN_GPUS,
    #         "allocation_cls": RayPerformanceAllocator,
    #         "cpu_ram_per_instance": 4,
    #         "trust_remote_code": True,
    #         "reasoning_parser": "qwen3",
    #         "default_chat_template_kwargs": {"enable_thinking": False},
    #     },
    # },
    {
        "model_id": "qwen-all-gpus",
        "description": "replicas=-1 (all eligible GPUs)",
        "config": {
            "model_name": "Qwen/Qwen3-0.6B-FP8",
            "max_input_prompt_length": 1024,
            "max_output_prompt_length": 2048,
            "replicas": -1,
            "cpu_ram_per_instance": -1,
            # HF model card / Qwen vLLM docs (enable-reasoning is deprecated; qwen3 since 0.9)
            "trust_remote_code": True,
            "reasoning_parser": "qwen3",
            "default_chat_template_kwargs": {"enable_thinking": False},
        },
    },
]

prompt = "Write a short poem about beer"
amount = 1_000
prompts = [f"{prompt} {i}" for i in range(amount)]
# Qwen3 non-thinking sampling (model card / deploy docs)
sample_kwargs = dict(max_tokens=100, temperature=0.0, top_p=0.8, top_k=20)

for idx, deployment in enumerate(deployments):
    model_id = deployment["model_id"]
    print(f"\n=== {deployment['description']} ({model_id}) ===")

    scheduler.deploy_model(model_id=model_id, **deployment["config"])
    time.sleep(2)

    _ = inference_batch(prompts[:10], model_id=model_id, **sample_kwargs)
    time.sleep(2)

    start = time.time()
    results = inference_batch(prompts, model_id=model_id, **sample_kwargs)
    elapsed = time.time() - start
    print(f"Processed {len(results)} prompts in {elapsed:.3f}s ({len(results)/elapsed:.2f} req/s)")

    scheduler.shutdown(model_id)
    if idx < len(deployments) - 1:
        time.sleep(3)
