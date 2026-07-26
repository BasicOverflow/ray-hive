"""Pinned same-node TP=2 — FP8 model split across two GPUs.

Qwen3-8B-FP8 (~8GB weights total → ~4GB/GPU at TP=2) on ergos-02 gpu1+gpu2.
7B bf16 (~14GB → ~7GB/GPU + overhead) does not fit the 8GB card here.
cpu_ram_per_instance=0 (GPU VRAM only).
"""
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.core.ray_utils import info, success
from ray_hive.inference import inference_batch

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

scheduler = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=False)

VLLM_KWARGS = dict(
    trust_remote_code=True,
    reasoning_parser="qwen3",
    default_chat_template_kwargs={"enable_thinking": False},
)

deployments = [
    {
        "model_id": "qwen3-8b-tp",
        "description": "Qwen3-8B-FP8 TP=2 on ergos-02 (~8GB → ~4GB/GPU)",
        "config": {
            "model_name": "Qwen/Qwen3-8B-FP8",
            "max_input_prompt_length": 512,
            "max_output_prompt_length": 512,
            "replicas": 1,
            "gpu": ["ergos-02-nv:gpu1", "ergos-02-nv:gpu2"],
            "cpu_ram_per_instance": 0,
            "vllm_kwargs": VLLM_KWARGS,
        },
    },
]

prompt = "Write a short poem about beer"
amount = 100
prompts = [f"{prompt} {i}" for i in range(amount)]
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
