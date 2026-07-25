"""Test OpenAI-compatible HTTP API (sync + streaming) against a deployed model."""
import json
import os
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.core.ray_utils import info
from ray_hive.inference import inference_stream

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_ID = "stream-demo"
MODEL_NAME = "Qwen/Qwen3-0.6B-FP8"
PROMPT = "Write a one-line joke about GPUs."
SAMPLE = dict(max_tokens=64, temperature=0.7, top_p=0.8, top_k=20)
MAX_IN, MAX_OUT, MAX_SEQS = 512, 256, 5


def serve_base_url() -> str:
    explicit = os.getenv("RAY_SERVE_URL")
    if explicit:
        return explicit.rstrip("/")
    host = os.environ["RAY_ADDRESS"].removeprefix("ray://").split(":")[0]
    return f"http://{host}:8000"


def request_json(method: str, path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{serve_base_url()}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def stream_openai(path: str, body: dict, text_key: str):
    """POST stream=true and print content/text deltas live from SSE."""
    req = urllib.request.Request(
        f"{serve_base_url()}{path}",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            choice = json.loads(data)["choices"][0]
            if text_key == "content":
                piece = (choice.get("delta") or {}).get("content") or ""
            else:
                piece = choice.get("text") or ""
            if piece:
                print(piece, end="", flush=True)
    print()


scheduler = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=False)
scheduler.estimate_vram(
    MODEL_NAME,
    max_num_seqs=MAX_SEQS,
    max_input_prompt_length=MAX_IN,
    max_output_prompt_length=MAX_OUT,
)
scheduler.deploy_model(
    model_id=MODEL_ID,
    model_name=MODEL_NAME,
    max_input_prompt_length=MAX_IN,
    max_output_prompt_length=MAX_OUT,
    max_num_seqs=MAX_SEQS,
    replicas=1,
    trust_remote_code=True,
    reasoning_parser="qwen3",
    default_chat_template_kwargs={"enable_thinking": False},
)

base = f"/{MODEL_ID}"

models = request_json("GET", f"{base}/v1/models")
info(f"GET /v1/models: {models}")

chat = request_json(
    "POST",
    f"{base}/v1/chat/completions",
    {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": PROMPT}],
        **SAMPLE,
    },
)
info(f"POST /v1/chat/completions: {chat['choices'][0]['message']['content']}")

completion = request_json(
    "POST",
    f"{base}/v1/completions",
    {"model": MODEL_ID, "prompt": PROMPT, **SAMPLE},
)
info(f"POST /v1/completions: {completion['choices'][0]['text']}")

info("inference_stream")
for delta in inference_stream(PROMPT, model_id=MODEL_ID, **SAMPLE):
    print(delta, end="", flush=True)
print()

info("OpenAI /v1/chat/completions stream")
stream_openai(
    f"{base}/v1/chat/completions",
    {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        **SAMPLE,
    },
    text_key="content",
)

info("OpenAI /v1/completions stream")
stream_openai(
    f"{base}/v1/completions",
    {"model": MODEL_ID, "prompt": PROMPT, "stream": True, **SAMPLE},
    text_key="text",
)

scheduler.shutdown(MODEL_ID)
