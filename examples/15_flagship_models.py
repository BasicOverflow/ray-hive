"""Reference deploys for current flagship open models.

Uncomment exactly one entry in `deployments` (or leave several to run
sequentially). Each block is a known-good Ray Hive shape for that checkpoint —
adjust context / replicas / gpu pins for your cluster.

Defaults prefer quantized / mid-size variants so a cold run is less likely to
OOM; larger dense flagships are commented with rough sizing notes.
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

load_dotenv(Path(__file__).resolve().parent / ".env")

scheduler = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=True)

# Optional same-node TP pin (edit host/gpus for your cluster).
TP2_GPUS = ["ergos-02-nv:gpu1", "ergos-02-nv:gpu2"]

QWEN3_KWARGS = dict(
    trust_remote_code=True,
    reasoning_parser="qwen3",
    default_chat_template_kwargs={"enable_thinking": False},
)
GEMMA4_KWARGS = dict(
    trust_remote_code=True,
    reasoning_parser="gemma4",
    default_chat_template_kwargs={"enable_thinking": False},
)
LLAMA4_KWARGS = dict(
    trust_remote_code=True,
)

deployments = [
    # --- practical / mid flagship (active) ---
    {
        "model_id": "gemma4-12b-w4a16",
        "description": "Gemma 4 12B QAT W4A16 — compact flagship text (auto-place)",
        "config": {
            "model_name": "google/gemma-4-12B-it-qat-w4a16-ct",
            "max_input_prompt_length": 2048,
            "max_output_prompt_length": 512,
            "replicas": 1,
            "vllm_kwargs": GEMMA4_KWARGS,
        },
    },
    # --- denser / larger flagships (uncomment one) ---
    # {
    #     "model_id": "qwen36-27b",
    #     "description": "Qwen3.6-27B dense (~bf16; often needs 80GB+ or TP)",
    #     "config": {
    #         "model_name": "Qwen/Qwen3.6-27B",
    #         "max_input_prompt_length": 4096,
    #         "max_output_prompt_length": 1024,
    #         "replicas": 1,
    #         "vllm_kwargs": QWEN3_KWARGS,
    #     },
    # },
    # {
    #     "model_id": "gemma4-31b",
    #     "description": "Gemma 4 31B IT (~bf16; typically 1×80GB or TP=2)",
    #     "config": {
    #         "model_name": "google/gemma-4-31B-it",
    #         "max_input_prompt_length": 4096,
    #         "max_output_prompt_length": 1024,
    #         "replicas": 1,
    #         # "gpu": TP2_GPUS,
    #         "vllm_kwargs": GEMMA4_KWARGS,
    #     },
    # },
    # {
    #     "model_id": "gemma4-26b-a4b",
    #     "description": "Gemma 4 26B-A4B MoE IT (~4B active; usually 1×80GB)",
    #     "config": {
    #         "model_name": "google/gemma-4-26B-A4B-it",
    #         "max_input_prompt_length": 4096,
    #         "max_output_prompt_length": 1024,
    #         "replicas": 1,
    #         "vllm_kwargs": GEMMA4_KWARGS,
    #     },
    # },
    # {
    #     "model_id": "llama4-scout",
    #     "description": "Llama 4 Scout 17B-16E Instruct (gated HF; often TP≥2)",
    #     "config": {
    #         "model_name": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    #         "max_input_prompt_length": 2048,
    #         "max_output_prompt_length": 512,
    #         "replicas": 1,
    #         "gpu": TP2_GPUS,
    #         "vllm_kwargs": LLAMA4_KWARGS,
    #     },
    # },
    # {
    #     "model_id": "qwen3-8b-fp8",
    #     "description": "Qwen3-8B-FP8 — smaller high-quality text baseline",
    #     "config": {
    #         "model_name": "Qwen/Qwen3-8B-FP8",
    #         "max_input_prompt_length": 4096,
    #         "max_output_prompt_length": 1024,
    #         "replicas": 1,
    #         "vllm_kwargs": QWEN3_KWARGS,
    #     },
    # },
]

prompt = "Write a short poem about beer"
amount = 20
prompts = [f"{prompt} {i}" for i in range(amount)]
sample_kwargs = dict(max_tokens=64, temperature=0.7, top_p=0.9)

for idx, deployment in enumerate(deployments):
    model_id = deployment["model_id"]
    cfg = deployment["config"]
    info(f"{deployment['description']} ({model_id})")

    scheduler.estimate_vram(**cfg)
    status = scheduler.deploy_model(model_id=model_id, **cfg)
    info(status)

    _ = inference_batch(prompts[:4], model_id=model_id, **sample_kwargs)
    start = time.time()
    results = inference_batch(prompts, model_id=model_id, **sample_kwargs)
    elapsed = time.time() - start
    success(
        f"Processed {len(results)} prompts in {elapsed:.3f}s "
        f"({len(results) / elapsed:.2f} req/s)"
    )

    scheduler.shutdown(model_id)
    if idx < len(deployments) - 1:
        time.sleep(3)
