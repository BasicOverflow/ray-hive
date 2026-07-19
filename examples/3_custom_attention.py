"""Deploy a model with externally defined attention specs for VRAM planning."""
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.core.model_specs import BaseAttentionSpecs
from ray_hive.core.ray_gpu_alloc import RayPerformanceAllocator
from ray_hive.inference import inference_batch

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# Example inheritance to calculate non-traditional attention.
class MQAAttentionSpecs(BaseAttentionSpecs):
    """KV cache calculator for multi-query attention models."""

    @property
    def kv_heads(self) -> int:
        """Return 1 — MQA uses a single shared KV head."""
        # MQA uses a single KV head shared across all query heads.
        return 1

    # NOTE: calc_max_num_seqs_given_kv_cache method stays the same, only difference is how kv_heads is calculated.
    # In Reality, Most MQA models expose:
    # {
    #   "num_attention_heads": 32,
    #   "num_key_value_heads": 1
    # }
    # In the hf config anyway, so the BaseAttentionSpecs class will already work for MQA models out of the box. However, this is just to demonstrate how to override a property from the base class.


# More custom attention inheritance examples:


class Qwen3AttentionSpecs(BaseAttentionSpecs):
    """
    KV cache calculator for Qwen3 (GQA + optional hybrid sliding-window layers).

    Qwen3 configs expose head_dim / num_key_value_heads directly. When
    use_sliding_window is set, layers i >= max_window_layers use sliding_attention
    (KV capped at sliding_window); earlier layers keep full seq-len KV.
    """

    @property
    def head_dim(self) -> int:
        """Return Qwen3 head_dim from config (always set on Qwen3)."""
        return self.hf_params["head_dim"]


    @property
    def kv_heads(self) -> int:
        """Return GQA KV heads from num_key_value_heads."""
        return self.hf_params["num_key_value_heads"]


    @property
    def num_layers(self) -> int:
        """Return transformer depth from num_hidden_layers."""
        return self.hf_params["num_hidden_layers"]


    def _layer_types(self) -> list[str]:
        """Return per-layer attention type list matching Qwen3Config semantics."""
        types = self.hf_params.get("layer_types")
        if types is not None:
            return list(types)

        n = self.num_layers
        if not self.hf_params.get("use_sliding_window"):
            return ["full_attention"] * n

        max_full = self.hf_params.get("max_window_layers", n)
        # HF: sliding_attention when sliding_window is set and i >= max_window_layers
        return [
            "sliding_attention" if i >= max_full else "full_attention"
            for i in range(n)
        ]


    def kv_bytes_per_sequence(self, max_model_len: int) -> float:
        """Return KV bytes for one sequence, accounting for sliding-window layers."""
        assert max_model_len > 0, f"max_model_len must be positive, got {max_model_len}"
        bytes_per_layer_token = (
            2 * self.kv_bytes_per_element * self.kv_heads * self.head_dim
        )
        window = self.hf_params.get("sliding_window")
        total = 0.0
        for layer_type in self._layer_types():
            if layer_type == "sliding_attention" and window is not None:
                seq_tokens = min(max_model_len, int(window))
            else:
                seq_tokens = max_model_len
            total += bytes_per_layer_token * seq_tokens
        assert total > 0, "kv_bytes_per_sequence must be positive"
        return total


scheduler = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=True)
model_id = "qwen-custom-attention"

scheduler.deploy_model(
    model_id=model_id,
    model_name="Qwen/Qwen3-0.6B-FP8",
    max_input_prompt_length=1024,
    max_output_prompt_length=2048,
    replicas=-1,
    attention_cls=Qwen3AttentionSpecs,
    allocation_cls=RayPerformanceAllocator,
    # HF model card / Qwen vLLM docs (enable-reasoning is deprecated; qwen3 since 0.9)
    trust_remote_code=True,
    reasoning_parser="qwen3",
    default_chat_template_kwargs={"enable_thinking": False},
)

prompt = "Write a short poem about beer"
amount = 10_000
prompts = [f"{prompt} {i}" for i in range(amount)]
# Qwen3 non-thinking sampling (model card / deploy docs)
sample_kwargs = dict(max_tokens=100, temperature=0.0, top_p=0.8, top_k=20)

time.sleep(2)
_ = inference_batch(prompts[:10], model_id=model_id, **sample_kwargs)

time.sleep(2)

start = time.time()
results = inference_batch(prompts, model_id=model_id, **sample_kwargs)
elapsed = time.time() - start
print(f"Processed {len(results)} prompts in {elapsed:.3f}s ({len(results)/elapsed:.2f} req/s)")

scheduler.shutdown(model_id)
