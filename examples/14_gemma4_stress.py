"""Stress-test Gemma 4 E2B on one maxed-out replica; timed batches per modality."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.core.model_specs import MultimodalAttentionSpecs
from ray_hive.core.ray_utils import file_to_data_url, info, success
from ray_hive.inference import inference_batch

load_dotenv(Path(__file__).resolve().parent / ".env")

MEDIA = Path(__file__).resolve().parent / "media"
IMAGE = MEDIA / "image_00.png"
AUDIO = MEDIA / "audio_00.wav"
VIDEO = MEDIA / "video_00.mp4"

MODEL_ID = "gemma4-stress"
MODEL_NAME = "google/gemma-4-E2B-it"
MAX_IN, MAX_OUT = 4096, 4096
AMOUNT = 1000



class Gemma4MultimodalAttentionSpecs(MultimodalAttentionSpecs):
    """Gemma 4 dual-attention KV (sliding/global head dims + KV sharing)."""

    def _layer_types(self) -> list[str]:
        types = self.hf_params.get("layer_types")
        if types is not None:
            return list(types)
        return ["full_attention"] * self.num_layers


    def _kv_producer_layer_types(self) -> list[str]:
        types = self._layer_types()
        shared = int(self.hf_params.get("num_kv_shared_layers") or 0)
        if shared <= 0:
            return types
        return types[: max(0, len(types) - shared)]


    def _layer_head_dim(self, layer_type: str) -> int:
        if layer_type == "full_attention":
            return int(self.hf_params.get("global_head_dim") or self.head_dim)
        return self.head_dim


    def _kv_tensors(self, layer_type: str) -> int:
        if layer_type == "full_attention" and self.hf_params.get("attention_k_eq_v"):
            return 1
        return 2


    @property
    def kv_layers(self) -> int:
        return len(self._kv_producer_layer_types())


    def kv_bytes_per_token(self) -> float:
        total = 0.0
        for layer_type in self._kv_producer_layer_types():
            total += (
                self._kv_tensors(layer_type)
                * self.kv_bytes_per_element
                * self.kv_heads
                * self._layer_head_dim(layer_type)
            )
        return total / self.tp_size


    def kv_bytes_per_sequence(self, max_model_len: int) -> float:
        assert max_model_len > 0, f"max_model_len must be positive, got {max_model_len}"
        window = self.hf_params.get("sliding_window")
        total = 0.0
        for layer_type in self._kv_producer_layer_types():
            bytes_per_token = (
                self._kv_tensors(layer_type)
                * self.kv_bytes_per_element
                * self.kv_heads
                * self._layer_head_dim(layer_type)
            )
            if layer_type == "sliding_attention" and window is not None:
                seq_tokens = min(max_model_len, int(window))
            else:
                seq_tokens = max_model_len
            total += bytes_per_token * seq_tokens
        return total / self.tp_size


def _bench(label: str, prompts: list, model_id: str, sample: dict):
    warm = max(1, min(8, len(prompts) // 4))
    _ = inference_batch(prompts[:warm], model_id=model_id, **sample)
    start = time.time()
    results = inference_batch(prompts, model_id=model_id, **sample)
    elapsed = time.time() - start
    success(f"{label}: {len(results)} req in {elapsed:.3f}s ({len(results) / elapsed:.2f} req/s)")


# No max_num_seqs — planner packs the single replica to fill VRAM.
VLLM_KWARGS = dict(
    trust_remote_code=True,
    reasoning_parser="gemma4",
    default_chat_template_kwargs={"enable_thinking": False},
    limit_mm_per_prompt={"image": 1, "audio": 1, "video": 1},
    mm_processor_kwargs={"max_soft_tokens": 280},
)

hive = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=True)
hive.estimate_vram(
    MODEL_NAME,
    max_input_prompt_length=MAX_IN,
    max_output_prompt_length=MAX_OUT,
    replicas=1,
    attention_cls=Gemma4MultimodalAttentionSpecs,
    vllm_kwargs=VLLM_KWARGS,
)
# status = hive.deploy_model(
#     model_id=MODEL_ID,
#     model_name=MODEL_NAME,
#     max_input_prompt_length=MAX_IN,
#     max_output_prompt_length=MAX_OUT,
#     replicas=1,
#     attention_cls=Gemma4MultimodalAttentionSpecs,
#     vllm_kwargs=VLLM_KWARGS,
# )
# info(status)

sample = dict(max_tokens=MAX_OUT, temperature=0.0)
img_url = file_to_data_url(IMAGE)
aud_url = file_to_data_url(AUDIO, "audio/wav")
vid_url = file_to_data_url(VIDEO, "video/mp4")

# _bench(
#     "text",
#     [f"Write a short poem about beer {i}" for i in range(AMOUNT)],
#     MODEL_ID,
#     sample,
# )
# _bench(
#     "image",
#     [[{
#         "role": "user",
#         "content": [
#             {"type": "image_url", "image_url": {"url": img_url}},
#             {"type": "text", "text": f"Write a short poem about this image {i}"},
#         ],
#     }] for i in range(AMOUNT)],
#     MODEL_ID,
#     sample,
# )
# _bench(
#     "audio",
#     [[{
#         "role": "user",
#         "content": [
#             {"type": "audio_url", "audio_url": {"url": aud_url}},
#             {"type": "text", "text": f"Write a short poem about this audio {i}"},
#         ],
#     }] for i in range(AMOUNT)],
#     MODEL_ID,
#     sample,
# )
# _bench(
#     "video",
#     [[{
#         "role": "user",
#         "content": [
#             {"type": "video_url", "video_url": {"url": vid_url}},
#             {"type": "text", "text": f"Write a short poem about this video {i}"},
#         ],
#     }] for i in range(AMOUNT)],
#     MODEL_ID,
#     sample,
# )

hive.shutdown(MODEL_ID)
