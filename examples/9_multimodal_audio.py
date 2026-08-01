"""Deploy an audio chat model and run inference with a wav fixture."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.core.ray_utils import file_to_data_url, info
from ray_hive.inference import inference

load_dotenv(Path(__file__).resolve().parent / ".env")

AUDIO = Path(__file__).resolve().parent / "media" / "audio_00.wav"
MODEL_ID = "audio-demo"
MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"
MAX_IN, MAX_OUT = 2048, 256


VLLM_KWARGS = dict(
    trust_remote_code=True,
    limit_mm_per_prompt={"audio": 1},
    max_num_seqs=2,
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
        {"type": "audio_url", "audio_url": {"url": file_to_data_url(AUDIO, "audio/wav")}},
        {"type": "text", "text": "What do you hear? One short sentence."},
    ],
}]
info(inference(messages, model_id=MODEL_ID, max_tokens=64))

hive.shutdown(MODEL_ID)
