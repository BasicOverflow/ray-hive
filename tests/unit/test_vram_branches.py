"""P — less common VRAM branches."""
from ray_hive.core.model_specs.planner import build_vram_reqs, normalize_hf_config


def test_moe_misc(tiny_hf_dense):
    hf = {**tiny_hf_dense, "num_experts": 8, "num_experts_per_tok": 2}
    vr = build_vram_reqs(hf)
    assert vr.calc_misc_vram_gb() >= 0


def test_hybrid_pattern(tiny_hf_dense):
    hf = {
        **tiny_hf_dense,
        "hybrid_override_pattern": "*-M",
        "mamba_num_heads": 4,
        "mamba_head_dim": 16,
        "num_hidden_layers": 3,
    }
    vr = build_vram_reqs(hf)
    assert vr.calc_weights_gb() > 0


def test_nested_text_config_mm(tiny_hf_dense):
    nested = {
        "model_type": "x",
        "text_config": tiny_hf_dense,
        "vision_config": {"hidden_size": 8},
    }
    flat = normalize_hf_config(nested)
    vr = build_vram_reqs(flat)
    assert vr.calc_weights_gb() > 0
