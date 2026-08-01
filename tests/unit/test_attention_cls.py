"""K — custom attention_cls changes plan."""
from ray_hive.core.model_specs.attention import BaseAttentionSpecs
from ray_hive.core.model_specs.planner import build_vram_reqs, plan_deployment


class FatKV(BaseAttentionSpecs):
    def kv_bytes_per_token(self) -> float:
        return super().kv_bytes_per_token() * 4


def test_custom_attention_changes_seqs(tiny_hf_dense):
    base = build_vram_reqs(tiny_hf_dense)
    fat = build_vram_reqs(tiny_hf_dense, attention_cls=FatKV)
    p_base = plan_deployment(
        base, 12.0, 24.0, 256, 128, 128,
    )
    p_fat = plan_deployment(
        fat, 12.0, 24.0, 256, 128, 128,
    )
    assert p_fat["max_num_seqs"] <= p_base["max_num_seqs"]
