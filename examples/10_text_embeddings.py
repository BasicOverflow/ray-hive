"""Deploy a text embedding model (runner=pooling) and fetch vectors via inference + HTTP."""
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.core.ray_utils import info, serve_base_url
from ray_hive.inference import inference, inference_batch

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL_ID = "embed-demo"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_IN = 512


VLLM_KWARGS = dict(
    runner="pooling",
    trust_remote_code=True,
    max_num_seqs=16,
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

vec = inference("Ray Hive embeds text.", model_id=MODEL_ID)
info(f"dim={len(vec)} head={vec[:4]}")

batch = inference_batch(["hello", "world"], model_id=MODEL_ID)
info(f"batch sizes={[len(v) for v in batch]}")

body = {"model": MODEL_ID, "input": ["OpenAI embeddings path"]}
req = urllib.request.Request(
    f"{serve_base_url()}/{MODEL_ID}/v1/embeddings",
    data=json.dumps(body).encode(),
    method="POST",
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.loads(resp.read().decode())
info(f"http dim={len(data['data'][0]['embedding'])}")

hive.shutdown(MODEL_ID)
