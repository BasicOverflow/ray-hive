"""Deploy/inference configs — 2 pinned replicas, CPU RAM spill/KV, and all GPUs."""
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
    # {
    #     "model_id": "qwen-two-replicas",
    #     "description": "replicas=2 (pinned)",
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
    {
        "model_id": "qwen-cpu-ram",
        "description": "replicas=2 + cpu_ram_per_instance=4",
        "config": {
            "model_name": "Qwen/Qwen3-0.6B-FP8",
            "max_input_prompt_length": 512,
            "max_output_prompt_length": 512,
            "replicas": -1,
            "cpu_ram_per_instance": 0,
            "vllm_kwargs": VLLM_KWARGS,
        },
    },
    # {
    #     "model_id": "qwen-all-gpus",
    #     "description": "replicas=-1 (all eligible GPUs)",
    #     "config": {
    #         "model_name": "Qwen/Qwen3-0.6B-FP8",
    #         "max_input_prompt_length": 512,
    #         "max_output_prompt_length": 512,
    #         "replicas": -1,
    #         "cpu_ram_per_instance": 2,
    #         "vllm_kwargs": VLLM_KWARGS,
    #     },
    # },
]

prompt = "Write a short poem about beer"
amount = 10_000
prompts = [f"{prompt} {i}" for i in range(amount)]
# Qwen3 non-thinking sampling (model card / deploy docs)
sample_kwargs = dict(max_tokens=100, temperature=0.0, top_p=0.8, top_k=20)

for idx, deployment in enumerate(deployments):
    model_id = deployment["model_id"]
    cfg = deployment["config"]
    info(f"{deployment['description']} ({model_id})")

    scheduler.estimate_vram(**cfg)
    status = scheduler.deploy_model(model_id=model_id, **cfg)
    info(status)

    _ = inference_batch(prompts[:10], model_id=model_id, **sample_kwargs)

    start = time.time()
    results = inference_batch(prompts, model_id=model_id, **sample_kwargs)
    elapsed = time.time() - start
    success(f"Processed {len(results)} prompts in {elapsed:.3f}s ({len(results)/elapsed:.2f} req/s)")

    scheduler.shutdown(model_id)
    if idx < len(deployments) - 1:
        time.sleep(3)
