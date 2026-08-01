"""Deploy a vision-language model and run image chat via inference + OpenAI HTTP."""
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.core.ray_utils import file_to_data_url, info, serve_base_url
from ray_hive.inference import inference

load_dotenv(Path(__file__).resolve().parent / ".env")

MEDIA = Path(__file__).resolve().parent / "media" / "image_00.png"
MODEL_ID = "vl-demo"
# Small VL model; swap for any vLLM-supported VL checkpoint + recommended kwargs.
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_IN, MAX_OUT = 2048, 256


VLLM_KWARGS = dict(
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 1},
    max_num_seqs=4,
)

hive = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=True)
hive.estimate_vram(
    MODEL_NAME,
    max_input_prompt_length=MAX_IN,
    max_output_prompt_length=MAX_OUT,
    vllm_kwargs=VLLM_KWARGS,
)

status = hive.deploy_model(
    model_id=MODEL_ID,
    model_name=MODEL_NAME,
    max_input_prompt_length=MAX_IN,
    max_output_prompt_length=MAX_OUT,
    replicas=1,
    vllm_kwargs=VLLM_KWARGS,
)
info(status)

messages = [{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": file_to_data_url(MEDIA)}},
        {"type": "text", "text": "Describe this image in one short sentence."},
    ],
}]
info(inference(messages, model_id=MODEL_ID, max_tokens=64))

body = {
    "model": MODEL_ID,
    "messages": messages,
    "max_tokens": 64,
}
req = urllib.request.Request(
    f"{serve_base_url()}/{MODEL_ID}/v1/chat/completions",
    data=json.dumps(body).encode(),
    method="POST",
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as resp:
    chat = json.loads(resp.read().decode())
info(chat["choices"][0]["message"]["content"])

hive.shutdown(MODEL_ID)
