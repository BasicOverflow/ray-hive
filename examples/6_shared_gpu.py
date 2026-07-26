"""Two single-replica models on one GPU: constrained first, then VRAM fill."""
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.core.ray_utils import info, success
from ray_hive.inference import inference_stream

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

scheduler = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=False)

PIN = "ergos-06-nv:gpu0"
MODEL_NAME = "Qwen/Qwen3-0.6B-FP8"
MAX_IN, MAX_OUT, MAX_SEQS = 512, 512, 8
PRINT_LOCK = threading.Lock()

BASE_KWARGS = dict(
    trust_remote_code=True,
    reasoning_parser="qwen3",
    default_chat_template_kwargs={"enable_thinking": False},
)

deployments = [
    {
        "model_id": "share-a",
        "description": f"constrained max_num_seqs={MAX_SEQS} on {PIN}",
        "prompt": "Write a short poem about beer.",
        "config": {
            "model_name": MODEL_NAME,
            "max_input_prompt_length": MAX_IN,
            "max_output_prompt_length": MAX_OUT,
            "replicas": 1,
            "gpu": PIN,
            "vllm_kwargs": {**BASE_KWARGS, "max_num_seqs": MAX_SEQS},
        },
    },
    {
        "model_id": "share-b",
        "description": f"max-out remaining VRAM on {PIN}",
        "prompt": "Write a short poem about GPUs.",
        "config": {
            "model_name": MODEL_NAME,
            "max_input_prompt_length": MAX_IN,
            "max_output_prompt_length": MAX_OUT,
            "replicas": 1,
            "gpu": PIN,
            "vllm_kwargs": dict(BASE_KWARGS),
        },
    },
]

sample_kwargs = dict(max_tokens=64, temperature=0.7, top_p=0.8, top_k=20)

for deployment in deployments:
    model_id = deployment["model_id"]
    cfg = deployment["config"]
    info(f"{deployment['description']} ({model_id})")
    scheduler.estimate_vram(**cfg)
    status = scheduler.deploy_model(model_id=model_id, **cfg)
    info(status)


def stream_one(model_id: str, prompt: str):
    for delta in inference_stream(prompt, model_id=model_id, **sample_kwargs):
        with PRINT_LOCK:
            print(f"[{model_id}]{delta}", end="", flush=True)
    with PRINT_LOCK:
        print(f"\n[{model_id}] done", flush=True)


info("Streaming share-a and share-b concurrently")
with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [
        pool.submit(stream_one, d["model_id"], d["prompt"])
        for d in deployments
    ]
    for f in futures:
        f.result()

success("Simultaneous streams finished")
for deployment in deployments:
    scheduler.shutdown(deployment["model_id"])
