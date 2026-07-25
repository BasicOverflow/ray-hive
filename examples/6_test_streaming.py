"""Test streaming via inference_stream (HTTP) and OpenAI-compatible SSE."""
import json
import os
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.inference import inference_stream

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_ID = "stream-demo"
MODEL_NAME = "Qwen/Qwen3-0.6B-FP8"
PROMPT = "Write a poem about beer."


def serve_base_url() -> str:
    explicit = os.getenv("RAY_SERVE_URL")
    if explicit:
        return explicit.rstrip("/")
    host = os.environ["RAY_ADDRESS"].removeprefix("ray://").split(":")[0]
    return f"http://{host}:8000"


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
scheduler.deploy_model(
    model_id=MODEL_ID,
    model_name=MODEL_NAME,
    max_input_prompt_length=512,
    max_output_prompt_length=256,
    max_num_seqs=5,
    replicas=1,
    trust_remote_code=True,
    reasoning_parser="qwen3",
    default_chat_template_kwargs={"enable_thinking": False},
)

sample = dict(max_tokens=128, temperature=0.7, top_p=0.8, top_k=20)

print("\n=== inference_stream (HTTP /v1/chat/completions) ===")
for delta in inference_stream(PROMPT, model_id=MODEL_ID, **sample):
    print(delta, end="", flush=True)
print()

print("\n=== OpenAI /v1/chat/completions stream ===")
stream_openai(
    f"/{MODEL_ID}/v1/chat/completions",
    {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        **sample,
    },
    text_key="content",
)

scheduler.shutdown(MODEL_ID)
