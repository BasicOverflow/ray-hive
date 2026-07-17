"""Test OpenAI-compatible HTTP API against a deployed model."""
import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_ID = os.getenv("MODEL_ID", "qwen-short-big-gpu")


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


base = f"/{MODEL_ID}"

models = request_json("GET", f"{base}/v1/models")
print("GET /v1/models:", models)

chat = request_json(
    "POST",
    f"{base}/v1/chat/completions",
    {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Write a one-line joke about GPUs."}],
        "max_tokens": 64,
        "temperature": 0.0,
    },
)
print("POST /v1/chat/completions:", chat["choices"][0]["message"]["content"])

completion = request_json(
    "POST",
    f"{base}/v1/completions",
    {
        "model": MODEL_ID,
        "prompt": "Ray is a",
        "max_tokens": 32,
        "temperature": 0.0,
    },
)
print("POST /v1/completions:", completion["choices"][0]["text"])
