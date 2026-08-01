"""Deploy a multimodal embedding model (VLM2Vec) and embed image+text."""
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
# vLLM pooling embed template for Phi-3.5-V based VLM2Vec (not the generate template).
CHAT_TEMPLATE = """\
{%- if messages | length > 1 -%}
    {{ raise_exception('Embedding models should only embed one message at a time') }}
{%- endif -%}

{% set vars = namespace(parts=[], next_image_id=1) %}
{%- for message in messages -%}
    {%- for content in message['content'] -%}
        {%- if content['type'] == 'text' -%}
            {%- set vars.parts = vars.parts + [content['text']] %}
        {%- elif content['type'] == 'image' -%}
            {%- set vars.parts = vars.parts + ['<|image_{i:d}|>'.format(i=vars.next_image_id)] %}
            {%- set vars.next_image_id = vars.next_image_id + 1 %}
        {%- endif -%}
    {%- endfor -%}
{%- endfor -%}
{{ vars.parts | join(' ') }}
"""
MODEL_ID = "mm-embed-demo"
MODEL_NAME = "TIGER-Lab/VLM2Vec-Full"
MAX_IN = 2048


VLLM_KWARGS = dict(
    runner="pooling",
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 1},
    mm_processor_kwargs={"num_crops": 4},
    chat_template=CHAT_TEMPLATE,
    max_num_seqs=4,
)

hive = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=True)
hive.estimate_vram(
    MODEL_NAME,
    max_input_prompt_length=MAX_IN,
    max_output_prompt_length=0,
    vllm_kwargs=VLLM_KWARGS,
)
status = hive.deploy_model(
    model_id=MODEL_ID,
    model_name=MODEL_NAME,
    max_input_prompt_length=MAX_IN,
    max_output_prompt_length=0,
    replicas=1,
    vllm_kwargs=VLLM_KWARGS,
)
info(status)

messages = [{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": file_to_data_url(MEDIA)}},
        {"type": "text", "text": "Represent the given image with the following question: a small red square"},
    ],
}]
vec = inference(messages, model_id=MODEL_ID)
info(f"dim={len(vec)} head={vec[:4]}")

body = {"model": MODEL_ID, "messages": messages}
req = urllib.request.Request(
    f"{serve_base_url()}/{MODEL_ID}/v1/embeddings",
    data=json.dumps(body).encode(),
    method="POST",
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as resp:
    data = json.loads(resp.read().decode())
info(f"http dim={len(data['data'][0]['embedding'])}")

hive.shutdown(MODEL_ID)
