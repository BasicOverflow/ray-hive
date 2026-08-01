"""Deploy Gemma 4 E2B and exercise text / image / audio / video with custom dual-attention KV sizing."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.core.model_specs import MultimodalAttentionSpecs
from ray_hive.core.ray_utils import file_to_data_url, info
from ray_hive.inference import inference

load_dotenv(Path(__file__).resolve().parent / ".env")

MEDIA = Path(__file__).resolve().parent / "media"
IMAGE = MEDIA / "image_00.png"
AUDIO = MEDIA / "audio_00.wav"
VIDEO = MEDIA / "video_00.mp4"

MODEL_ID = "gemma4-mm"
# Smallest Gemma 4 IT with native text + image + audio (video = frame sequence).
MODEL_NAME = "google/gemma-4-E2B-it"
MAX_IN, MAX_OUT = 4096, 128


class Gemma4MultimodalAttentionSpecs(MultimodalAttentionSpecs):
    """
    Gemma 4 dual-attention KV sizing (E2B/E4B/12B/31B HF configs).

    Architecture (see google/gemma-4-E2B):
    - Hybrid local/global: layer_types interleave sliding_attention and
      full_attention (E2B is 4:1 local:global; last layer always global).
    - Sliding layers: head_dim + KV capped at sliding_window (512 on E2B).
    - Global layers: global_head_dim (often 2× head_dim) + full seq len.
    - KV sharing: last num_kv_shared_layers reuse earlier KV (E2B: 20/35 →
      only 15 producer layers store cache).
    - Optional attention_k_eq_v on global layers (larger Gemma 4; E2B/E4B False).
    - MQA/GQA via num_key_value_heads (E2B: 1).
    """

    def _layer_types(self) -> list[str]:
        types = self.hf_params.get("layer_types")
        if types is not None:
            return list(types)
        return ["full_attention"] * self.num_layers


    def _kv_producer_layer_types(self) -> list[str]:
        """Layers that own a KV cache slot (excludes trailing shared layers)."""
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
        """1 if global layers reuse K as V, else 2 (K and V)."""
        if layer_type == "full_attention" and self.hf_params.get("attention_k_eq_v"):
            return 1
        return 2


    @property
    def kv_layers(self) -> int:
        """Unique KV-producing layers after sharing."""
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


VLLM_KWARGS = dict(
    trust_remote_code=True,
    reasoning_parser="gemma4",
    default_chat_template_kwargs={"enable_thinking": False},
    limit_mm_per_prompt={"image": 1, "audio": 1, "video": 1},
    mm_processor_kwargs={"max_soft_tokens": 280},
    max_num_seqs=2,
)

hive = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=True)
hive.estimate_vram(
    MODEL_NAME,
    max_input_prompt_length=MAX_IN,
    max_output_prompt_length=MAX_OUT,
    attention_cls=Gemma4MultimodalAttentionSpecs,
    vllm_kwargs=VLLM_KWARGS,
)
status = hive.deploy_model(
    model_id=MODEL_ID,
    model_name=MODEL_NAME,
    max_input_prompt_length=MAX_IN,
    max_output_prompt_length=MAX_OUT,
    replicas=1,
    attention_cls=Gemma4MultimodalAttentionSpecs,
    vllm_kwargs=VLLM_KWARGS,
)
info(status)

sample = dict(max_tokens=64, temperature=0.0)

info(inference("Say hello in one short sentence.", model_id=MODEL_ID, **sample))

info(inference([{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": file_to_data_url(IMAGE)}},
        {"type": "text", "text": "Describe this image in one short sentence."},
    ],
}], model_id=MODEL_ID, **sample))

info(inference([{
    "role": "user",
    "content": [
        {"type": "audio_url", "audio_url": {"url": file_to_data_url(AUDIO, "audio/wav")}},
        {"type": "text", "text": "What do you hear? One short sentence."},
    ],
}], model_id=MODEL_ID, **sample))

info(inference([{
    "role": "user",
    "content": [
        {"type": "video_url", "video_url": {"url": file_to_data_url(VIDEO, "video/mp4")}},
        {"type": "text", "text": "Describe this video in one short sentence."},
    ],
}], model_id=MODEL_ID, **sample))

hive.shutdown(MODEL_ID)
