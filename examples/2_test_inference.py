"""Test OpenAI-compatible HTTP API (sync + streaming) and LangChain against a deployed model."""
import json
import os
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from ray_hive import RayHive
from ray_hive.core.ray_utils import info
from ray_hive.inference import inference_stream

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL_ID = "stream-demo"
MODEL_NAME = "Qwen/Qwen3-0.6B-FP8"
PROMPT = "Write a one-line joke about GPUs."
SAMPLE = dict(max_tokens=64, temperature=0.7, top_p=0.8, top_k=20)
MAX_IN, MAX_OUT, MAX_SEQS = 512, 256, 5


class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")


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


VLLM_KWARGS = dict(
    max_num_seqs=MAX_SEQS,
    trust_remote_code=True,
    reasoning_parser="qwen3",
    default_chat_template_kwargs={"enable_thinking": False},
)

scheduler = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=True)
scheduler.estimate_vram(
    MODEL_NAME,
    max_input_prompt_length=MAX_IN,
    max_output_prompt_length=MAX_OUT,
    vllm_kwargs=VLLM_KWARGS,
)
status = scheduler.deploy_model(
    model_id=MODEL_ID,
    model_name=MODEL_NAME,
    max_input_prompt_length=MAX_IN,
    max_output_prompt_length=MAX_OUT,
    replicas=1,
    vllm_kwargs=VLLM_KWARGS,
)
info(status)

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

llm = ChatOpenAI(
    base_url=f"{serve_base_url()}{base}/v1",
    api_key="EMPTY",
    model=MODEL_ID,
    max_tokens=SAMPLE["max_tokens"],
    temperature=SAMPLE["temperature"],
    extra_body={"top_p": SAMPLE["top_p"], "top_k": SAMPLE["top_k"]},
)

info("LangChain ChatOpenAI")
info(llm.invoke(PROMPT).content)

info("LangChain with_structured_output")
info(llm.with_structured_output(Joke).invoke("Tell a short joke about GPUs"))

scheduler.shutdown(MODEL_ID)
