"""Exercise sleep_timeout → wake via OpenAI HTTP → idle_timeout self-destroy.

Uses HTTP for request traffic so Ray Client → Serve handle lag cannot race the
short sleep/idle timers (GCS stalls of 30s+ would otherwise idle-destroy first).
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.core.ray_utils import info, success

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL_ID = "sleep-demo"
MODEL_NAME = "Qwen/Qwen3-0.6B-FP8"
SLEEP_TIMEOUT = 10
IDLE_TIMEOUT = 30
SAMPLE = dict(max_tokens=32, temperature=0.7, top_p=0.8, top_k=20)
VLLM_KWARGS = dict(
    max_num_seqs=5,
    trust_remote_code=True,
    reasoning_parser="qwen3",
    default_chat_template_kwargs={"enable_thinking": False},
)


def serve_base_url() -> str:
    explicit = os.getenv("RAY_SERVE_URL")
    if explicit:
        return explicit.rstrip("/")
    host = os.environ["RAY_ADDRESS"].removeprefix("ray://").split(":")[0]
    return f"http://{host}:8000"


def openai_chat(prompt: str) -> str:
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        **SAMPLE,
    }
    req = urllib.request.Request(
        f"{serve_base_url()}/{MODEL_ID}/v1/chat/completions",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


scheduler = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=True)

info(f"Deploy {MODEL_ID} sleep={SLEEP_TIMEOUT}s idle={IDLE_TIMEOUT}s")
status = scheduler.deploy_model(
    model_id=MODEL_ID,
    model_name=MODEL_NAME,
    max_input_prompt_length=512,
    max_output_prompt_length=256,
    replicas=1,
    sleep_timeout=SLEEP_TIMEOUT,
    idle_timeout=IDLE_TIMEOUT,
    vllm_kwargs=VLLM_KWARGS,
)
info(status)

info("Hot: OpenAI chat completion")
print(openai_chat("Say hello in one short sentence."))

wait_sleep = SLEEP_TIMEOUT + 5
info(f"Waiting {wait_sleep}s for sleep_timeout...")
time.sleep(wait_sleep)

info("Wake via OpenAI HTTP after sleep")
print(openai_chat("One word: awake"))

wait_idle = IDLE_TIMEOUT + 5
info(f"Waiting {wait_idle}s for idle_timeout (no requests)...")
time.sleep(wait_idle)

info("Checking model is gone after idle destroy")
try:
    openai_chat("Should fail")
    raise RuntimeError("model still responding after idle_timeout")
except urllib.error.HTTPError as e:
    success(f"OpenAI HTTP failed as expected: {e.code}")
except Exception as e:
    success(f"Request failed as expected: {type(e).__name__}: {e}")

success("sleep → wake → idle destroy cascade ok")
