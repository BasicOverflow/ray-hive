"""Deploy/inference configs — estimate_vram, pin, replicas=-1, auto-place.

Commented templates below (swap into `deployments` to run):
  - replicas=2 with gpu=[...]  → N single-GPU pins
  - replicas=-1               → every eligible GPU
Active entry is auto-placed replicas=1.
"""
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.core.ray_gpu_alloc import RayPerformanceAllocator
from ray_hive.core.ray_utils import info, success
from ray_hive.inference import inference_batch

load_dotenv(Path(__file__).resolve().parent / ".env")

scheduler = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=True)

PIN_GPUS = ["ergos-06-nv:gpu0", "ergos-04-nv:gpu0"]
VLLM_KWARGS = dict(
    trust_remote_code=True,
    reasoning_parser="qwen3",
    default_chat_template_kwargs={"enable_thinking": False},
)

deployments = [
    # Template: N single-GPU pins (replicas == len(gpu))
    # {
    #     "model_id": "qwen-two-replicas",
    #     "description": "replicas=2 pinned to PIN_GPUS (TP=1 each)",
    #     "config": {
    #         "model_name": "Qwen/Qwen3-0.6B-FP8",
    #         "max_input_prompt_length": 512,
    #         "max_output_prompt_length": 512,
    #         "replicas": 2,
    #         "gpu": PIN_GPUS,
    #         "allocation_cls": RayPerformanceAllocator,
    #         "vllm_kwargs": VLLM_KWARGS,
    #     },
    # },
    # Template: every eligible GPU under default allocation policy
    {
        "model_id": "qwen-all-gpus",
        "description": "replicas=-1 (all eligible GPUs)",
        "config": {
            "model_name": "Qwen/Qwen3-0.6B-FP8",
            "max_input_prompt_length": 4096,
            "max_output_prompt_length": 4096,
            "replicas": 1,
            "vllm_kwargs": VLLM_KWARGS,
        },
    },
    # {
    #     "model_id": "gemma4-12b-w4a16",
    #     "description": "Gemma 4 12B QAT W4A16 auto-placed",
    #     "config": {
    #         "model_name": "google/gemma-4-12B-it-qat-w4a16-ct",
    #         "max_input_prompt_length": 512,
    #         "max_output_prompt_length": 512,
    #         "replicas": 1,
    #         "vllm_kwargs": dict(
    #             trust_remote_code=True,
    #             reasoning_parser="gemma4",
    #             default_chat_template_kwargs={"enable_thinking": False},
    #         ),
    #     },
    # },
]

prompt = "Write a short poem about beer"
amount = 1_000
prompts = [f"{prompt} {i}" for i in range(amount)]
sample_kwargs = dict(max_tokens=100, temperature=1.0, top_p=0.95, top_k=64)

for idx, deployment in enumerate(deployments):
    model_id = deployment["model_id"]
    cfg = deployment["config"]
    info(f"{deployment['description']} ({model_id})")

    scheduler.estimate_vram(**cfg)
    status = scheduler.deploy_model(model_id=model_id, **cfg)
    info(status)

    # _ = inference_batch(prompts[:10], model_id=model_id, **sample_kwargs)

    # start = time.time()
    # results = inference_batch(prompts, model_id=model_id, **sample_kwargs)
    # elapsed = time.time() - start
    # success(f"Processed {len(results)} prompts in {elapsed:.3f}s ({len(results)/elapsed:.2f} req/s)")

    # scheduler.shutdown(model_id)
    # if idx < len(deployments) - 1:
    #     time.sleep(3)
