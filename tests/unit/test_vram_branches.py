"""P — less common VRAM branches."""
import pytest

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


def test_sleep_peak_factor_scales_fixed_non_kv(tiny_hf_dense):
    base = build_vram_reqs(tiny_hf_dense, sleep_peak_factor=1.0)
    full = base.calc_fixed_non_kv_gb(sleep_mode=True)
    none = build_vram_reqs(tiny_hf_dense, sleep_peak_factor=0.0).calc_fixed_non_kv_gb(
        sleep_mode=True
    )
    half = build_vram_reqs(tiny_hf_dense, sleep_peak_factor=0.5).calc_fixed_non_kv_gb(
        sleep_mode=True
    )
    awake = base.calc_fixed_non_kv_gb(sleep_mode=False)
    assert none == pytest.approx(awake)
    assert full > awake
    assert half == pytest.approx(awake + 0.5 * (full - awake))


def test_custom_vram_cls_override(tiny_hf_dense):
    from ray_hive.core.model_specs.vram_reqs import BaseVramReqs

    class ZeroSleep(BaseVramReqs):
        def calc_sleep_peak_gb(self, sleep_mode: bool = False) -> float:
            return 0.0

    default = build_vram_reqs(tiny_hf_dense, sleep_peak_factor=1.0).calc_fixed_non_kv_gb(sleep_mode=True)
    custom = build_vram_reqs(tiny_hf_dense, vram_cls=ZeroSleep).calc_fixed_non_kv_gb(
        sleep_mode=True
    )
    awake = build_vram_reqs(tiny_hf_dense).calc_fixed_non_kv_gb(sleep_mode=False)
    assert custom == pytest.approx(awake)
    assert default > custom



def test_text_only_drops_vision_from_checkpoint_floor(tiny_hf_mm):
    hf = {**tiny_hf_mm, "_checkpoint_bytes": int(8 * 1024**3)}
    with_vision = build_vram_reqs(hf).calc_weights_gb()
    text_only = build_vram_reqs(hf, language_model_only=True).calc_weights_gb()
    assert text_only < with_vision


def test_mtp_draft_adds_lm_head(tiny_hf_dense):
    base = build_vram_reqs(tiny_hf_dense)
    mtp = build_vram_reqs(
        tiny_hf_dense,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
    )
    assert base.calc_draft_weights_gb() == 0.0
    assert mtp.calc_draft_weights_gb() > 0.0
    assert mtp.calc_fixed_non_kv_gb() > base.calc_fixed_non_kv_gb()
    assert mtp.attention.kv_bytes_per_token() == base.attention.kv_bytes_per_token()


def test_dflash_uses_checkpoint_not_embed_formula(tiny_hf_dense):
    """DFlash2 shares target embed/lm_head; packed 1.19 GiB must win over vocab math."""
    draft = {
        **tiny_hf_dense,
        "num_hidden_layers": 5,
        "vocab_size": 248320,
        "hidden_size": 5120,
        "_checkpoint_bytes": int(1.19 * 1024**3),
    }
    spec = build_vram_reqs(
        tiny_hf_dense,
        speculative_config={
            "method": "dflash",
            "model": "draft/dflash",
            "num_speculative_tokens": 7,
        },
        _draft_hf=draft,
    )
    assert spec.calc_draft_weights_gb() == pytest.approx(1.19 * 1.10, rel=1e-6)
    # Formula path would have counted a 2.4 GiB unused vocab table.
    assert spec.calc_draft_weights_gb() < 2.0


def test_separate_draft_adds_weights_and_kv(tiny_hf_dense):
    draft = {**tiny_hf_dense, "num_hidden_layers": 4, "vocab_size": 512}
    base = build_vram_reqs(tiny_hf_dense)
    spec = build_vram_reqs(
        tiny_hf_dense,
        speculative_config={"method": "eagle", "model": "draft/dummy"},
        _draft_hf=draft,
    )
    assert spec.calc_draft_weights_gb() > 0.0
    assert spec.attention.kv_bytes_per_token() > base.attention.kv_bytes_per_token()


def test_nested_text_config_mm(tiny_hf_dense):
    nested = {
        "model_type": "x",
        "text_config": tiny_hf_dense,
        "vision_config": {"hidden_size": 8},
    }
    flat = normalize_hf_config(nested)
    vr = build_vram_reqs(flat)
    assert vr.calc_weights_gb() > 0
