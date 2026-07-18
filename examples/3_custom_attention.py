"""Deploy a model with externally defined attention specs for VRAM planning."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.core.model_specs import BaseAttentionSpecs
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


class Qwen35AttentionSpecs(BaseAttentionSpecs):
    """
    KV + linear-state calculator for Qwen3.5 hybrid attention.

    Full-attention layers: classic seq-len KV cache.
    Linear-attention (GatedDeltaNet) layers: fixed-size conv + recurrent state per sequence.
    """

    def _layer_types(self) -> list[str]:
        types = self.hf_params.get("layer_types")
        if types is not None:
            return list(types)
 
        interval = self.hf_params.get("full_attention_interval")
        if interval is not None:
            return [
                "linear_attention" if (i + 1) % interval else "full_attention"
                for i in range(self.num_layers)
            ]

        if self.hf_params.get("num_attention_layers") is not None:
            n_full = self.hf_params["num_attention_layers"]
            return ["full_attention"] * n_full + ["linear_attention"] * (self.num_layers - n_full)

        return ["full_attention"] * self.num_layers


    @property
    def kv_layers(self) -> int:
        """Return count of full-attention layers that hold a seq-len KV cache."""
        return sum(1 for t in self._layer_types() if t == "full_attention")


    @property
    def linear_layers(self) -> int:
        """Return count of GatedDeltaNet / linear-attention layers."""
        return sum(1 for t in self._layer_types() if t == "linear_attention")


    def _linear_state_dtype_bytes(self) -> float:
        """Return bytes per element for GatedDeltaNet state (usually fp32)."""
        dtype_sizes = {
            "float32": 4, "fp32": 4, "float": 4,
            "bfloat16": 2, "bf16": 2,
            "float16": 2, "fp16": 2, "half": 2,
        }
        name = str(self.hf_params.get("mamba_ssm_dtype", "float32")).lower().split(".")[-1]
        return dtype_sizes.get(name, 4)


    def linear_state_bytes_per_sequence(self) -> float:
        """Return fixed GatedDeltaNet conv+recurrent state bytes for one sequence."""
        value_heads = self.hf_params["linear_num_value_heads"]
        key_dim = self.hf_params["linear_key_head_dim"]
        value_dim = self.hf_params["linear_value_head_dim"]
        conv_dim = self.hf_params["linear_conv_kernel_dim"]
        b = self._linear_state_dtype_bytes()

        # conv_state ≈ (d_inner, d_conv); recurrent ≈ (value_heads, value_dim, key_dim)
        d_inner = value_heads * value_dim
        conv = d_inner * conv_dim * b
        recurrent = value_heads * value_dim * key_dim * b
        return self.linear_layers * (conv + recurrent)


    def kv_bytes_per_sequence(self, max_model_len: int) -> float:
        """Return full-attn KV (scales with len) + linear state (fixed) for one sequence."""
        return (
            self.kv_bytes_per_token() * max_model_len
            + self.linear_state_bytes_per_sequence()
        )


scheduler = RayHive(suppress_logging=True)
model_id = "qwen35-custom-attention"

scheduler.deploy_model(
    model_id=model_id,
    model_name="vadery/Qwen3.5-0.8B-W8A8",
    max_input_prompt_length=1024,
    max_output_prompt_length=2048,
    replicas=-1,
    attention_cls=Qwen35AttentionSpecs,
    trust_remote_code=True,
)

prompt = "Write a short poem about beer"
amount = 10_000
prompts = [f"{prompt} {i}" for i in range(amount)]

time.sleep(2)
_ = inference_batch(prompts, model_id=model_id, max_tokens=100, temperature=0.0)

time.sleep(2)

start = time.time()
results = inference_batch(prompts, model_id=model_id, max_tokens=100, temperature=0.0)
elapsed = time.time() - start
print(f"Processed {len(results)} prompts in {elapsed:.3f}s ({len(results)/elapsed:.2f} req/s)")

scheduler.shutdown(model_id)
