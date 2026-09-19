"""Auto max_input_prompt_length packing (fixed max_num_seqs)."""
import pytest

from ray_hive.core.model_specs.attention import BaseAttentionSpecs
from ray_hive.core.model_specs.planner import build_vram_reqs, effective_input_len
from ray_hive.core.ray_utils.placement import (
    AUTO_HYBRID_INPUT_CAP,
    AUTO_INPUT_FLOOR,
    AUTO_INPUT_MIN,
    is_auto_input,
    plan_replica_groups,
    solve_auto_text_input,
    validate_auto_input_config,
)
from ray_hive.errors import ConfigError, KvBudgetError, ModelDoesNotFitError
from tests.helpers import FakePerformanceAllocator, make_gpu


class FatKV(BaseAttentionSpecs):
    """Inflate KV so auto context is VRAM-bound on tiny fixtures."""

    def kv_bytes_per_token(self) -> float:
        return super().kv_bytes_per_token() * 64


def _pin_config(**overrides):
    cfg = {
        "name": "tiny",
        "replicas": 1,
        "gpu": "host-a:gpu0",
        "max_output_prompt_length": 128,
        "allocation_cls": FakePerformanceAllocator,
        "attention_cls": FatKV,
        "max_num_seqs": 8,
    }
    cfg.update(overrides)
    return cfg


def test_is_auto_input():
    assert is_auto_input("auto")
    assert not is_auto_input("AUTO")
    assert not is_auto_input(256)


def test_auto_requires_max_num_seqs():
    with pytest.raises(ConfigError, match="max_num_seqs"):
        validate_auto_input_config({"max_input_prompt_length": "auto"})


def test_auto_rejects_bad_max_num_seqs():
    with pytest.raises(ConfigError, match="max_num_seqs"):
        validate_auto_input_config({"max_input_prompt_length": "auto", "max_num_seqs": 0})


def test_auto_input_floor_and_surfaces(tiny_hf_dense):
    gmap = {"host-a:gpu0": make_gpu("host-a:gpu0", 10.0, 12.0)}
    results = plan_replica_groups(
        gmap,
        _pin_config(max_input_prompt_length="auto"),
        tiny_hf_dense,
        {},
        model_id="m",
    )
    assert len(results) == 1
    plan = next(iter(results.values()))["plan"]
    assert plan["max_input_prompt_length"] >= AUTO_INPUT_FLOOR
    assert plan["max_input_prompt_length_auto"] is True
    assert plan["max_output_prompt_length"] == 128
    assert plan["max_num_seqs"] == 8
    assert plan["max_model_len"] == plan["max_input_prompt_length"] + 128
    assert next(iter(results.values()))["max_model_len"] == plan["max_model_len"]


def test_auto_grows_with_more_vram(tiny_hf_dense):
    small = plan_replica_groups(
        {"host-a:gpu0": make_gpu("host-a:gpu0", 3.0, 4.0)},
        _pin_config(max_input_prompt_length="auto", max_num_seqs=16),
        tiny_hf_dense,
        {},
        model_id="s",
    )
    large = plan_replica_groups(
        {"host-a:gpu0": make_gpu("host-a:gpu0", 20.0, 24.0)},
        _pin_config(max_input_prompt_length="auto", max_num_seqs=16),
        tiny_hf_dense,
        {},
        model_id="l",
    )
    s_in = next(iter(small.values()))["plan"]["max_input_prompt_length"]
    l_in = next(iter(large.values()))["plan"]["max_input_prompt_length"]
    assert l_in > s_in
    assert s_in >= AUTO_INPUT_FLOOR


def test_auto_respects_hf_position_cap(tiny_hf_dense):
    hf = {**tiny_hf_dense, "max_position_embeddings": 512}
    # Without FatKV the tiny model would smash the abs cap; still cap by HF.
    gmap = {"host-a:gpu0": make_gpu("host-a:gpu0", 22.0, 24.0)}
    cfg = _pin_config(max_input_prompt_length="auto", attention_cls=None, max_num_seqs=2)
    results = plan_replica_groups(gmap, cfg, hf, {}, model_id="cap")
    plan = next(iter(results.values()))["plan"]
    # text + output <= 512
    assert plan["max_input_prompt_length"] + plan["max_output_prompt_length"] <= 512
    assert plan["max_model_len"] <= 512


def test_fixed_input_still_annotated(tiny_hf_dense):
    gmap = {"host-a:gpu0": make_gpu("host-a:gpu0", 10.0, 12.0)}
    results = plan_replica_groups(
        gmap,
        _pin_config(max_input_prompt_length=1024, max_num_seqs=4, attention_cls=None),
        tiny_hf_dense,
        {},
        model_id="fixed",
    )
    plan = next(iter(results.values()))["plan"]
    assert plan["max_input_prompt_length"] == 1024
    assert plan["max_input_prompt_length_auto"] is False
    assert plan["max_model_len"] == 1024 + 128


def test_auto_mm_includes_placeholders(tiny_hf_mm):
    gmap = {"host-a:gpu0": make_gpu("host-a:gpu0", 16.0, 24.0)}
    vllm = {"limit_mm_per_prompt": {"image": 1, "audio": 0}}
    results = plan_replica_groups(
        gmap,
        _pin_config(max_input_prompt_length="auto", max_num_seqs=4, attention_cls=None),
        tiny_hf_mm,
        vllm,
        model_id="mm",
    )
    entry = next(iter(results.values()))
    plan = entry["plan"]
    assert plan["max_input_prompt_length"] >= AUTO_INPUT_FLOOR
    assert plan["mm_tokens_per_prompt"] > 0
    # Effective context = text + MM; engine max_model_len must cover placeholders + output.
    vr = build_vram_reqs(tiny_hf_mm, **vllm)
    effective = effective_input_len(vr, plan["max_input_prompt_length"])
    assert effective == plan["max_input_prompt_length"] + plan["mm_tokens_per_prompt"]
    assert entry["max_model_len"] == effective + plan["max_output_prompt_length"]
    assert entry["max_model_len"] >= plan["mm_tokens_per_prompt"] + plan["max_output_prompt_length"]


def test_auto_mm_hf_cap_reserves_placeholders(tiny_hf_mm):
    # Soft-token image budget so HF window has room for text floor + output.
    hf = {**tiny_hf_mm, "max_position_embeddings": 1024}
    gmap = {"host-a:gpu0": make_gpu("host-a:gpu0", 22.0, 24.0)}
    vllm = {
        "limit_mm_per_prompt": {"image": 1, "audio": 0},
        "mm_processor_kwargs": {"max_soft_tokens": 128},
    }
    results = plan_replica_groups(
        gmap,
        _pin_config(
            max_input_prompt_length="auto",
            max_output_prompt_length=128,
            max_num_seqs=2,
            attention_cls=None,
        ),
        hf,
        vllm,
        model_id="mm-cap",
    )
    plan = next(iter(results.values()))["plan"]
    assert plan["mm_tokens_per_prompt"] == 128
    assert plan["max_model_len"] <= 1024
    assert plan["max_model_len"] == (
        plan["max_input_prompt_length"] + plan["mm_tokens_per_prompt"] + 128
    )
    assert plan["max_input_prompt_length"] + 128 + 128 <= 1024


class HugeKV(BaseAttentionSpecs):
    """~2 MiB/token so 256-token auto floor misses a tight 1-seq budget."""

    def kv_bytes_per_token(self) -> float:
        return 2 * 1024 * 1024


def test_auto_seqs_one_shrinks_below_floor(tiny_hf_dense):
    gmap = {"host-a:gpu0": make_gpu("host-a:gpu0", 0.85, 1.0)}
    results = plan_replica_groups(
        gmap,
        _pin_config(
            max_input_prompt_length="auto",
            max_num_seqs=1,
            max_output_prompt_length=128,
            attention_cls=HugeKV,
        ),
        tiny_hf_dense,
        {},
        model_id="shrink",
    )
    plan = next(iter(results.values()))["plan"]
    assert plan["max_num_seqs"] == 1
    assert plan["max_input_prompt_length_auto"] is True
    assert AUTO_INPUT_MIN <= plan["max_input_prompt_length"] < AUTO_INPUT_FLOOR


def test_auto_seqs_above_one_does_not_shrink(tiny_hf_dense):
    gmap = {"host-a:gpu0": make_gpu("host-a:gpu0", 0.85, 1.0)}
    with pytest.raises((ModelDoesNotFitError, KvBudgetError, ValueError)):
        plan_replica_groups(
            gmap,
            _pin_config(
                max_input_prompt_length="auto",
                max_num_seqs=8,
                max_output_prompt_length=128,
                attention_cls=HugeKV,
            ),
            tiny_hf_dense,
            {},
            model_id="noshrink",
        )


def test_auto_floor_does_not_fit_raises(tiny_hf_dense):
    # Near-zero free VRAM after weights → floor cannot pack.
    gmap = {"host-a:gpu0": make_gpu("host-a:gpu0", 0.05, 24.0)}
    with pytest.raises((ModelDoesNotFitError, ValueError, ConfigError)):
        plan_replica_groups(
            gmap,
            _pin_config(max_input_prompt_length="auto", max_num_seqs=32),
            tiny_hf_dense,
            {},
            model_id="oom",
        )


def test_solve_auto_text_input_direct(tiny_hf_dense):
    vr = build_vram_reqs(tiny_hf_dense, attention_cls=FatKV)
    plan, text_in, input_len, mml = solve_auto_text_input(
        vr,
        tiny_hf_dense,
        text_out=64,
        vram_budget_gb=8.0,
        live_total_vram_gb=12.0,
        live_available_vram_gb=10.0,
        max_num_seqs=4,
    )
    assert text_in >= AUTO_INPUT_FLOOR
    assert input_len == text_in
    assert mml == text_in + 64
    assert plan["max_num_seqs"] == 4


def test_auto_hybrid_caps_overgrown_kv_window(tiny_hf_dense):
    """Cheap KV (few full-attn layers) must not auto-grow past the hybrid cap."""
    hf = {
        **tiny_hf_dense,
        "num_hidden_layers": 8,
        "layer_types": ["linear_attention"] * 7 + ["full_attention"],
        "max_position_embeddings": 32768,
    }
    gmap = {"host-a:gpu0": make_gpu("host-a:gpu0", 22.0, 24.0)}
    results = plan_replica_groups(
        gmap,
        _pin_config(
            max_input_prompt_length="auto",
            max_num_seqs=1,
            max_output_prompt_length=256,
            attention_cls=None,
        ),
        hf,
        {},
        model_id="hybrid-cap",
    )
    plan = next(iter(results.values()))["plan"]
    assert plan["max_input_prompt_length"] <= AUTO_HYBRID_INPUT_CAP
    assert plan["max_input_prompt_length"] >= AUTO_INPUT_FLOOR


def test_auto_hybrid_cap_override_grows_window(tiny_hf_dense):
    """Raising auto_hybrid_input_cap lets leftover VRAM become context."""
    hf = {
        **tiny_hf_dense,
        "num_hidden_layers": 8,
        "layer_types": ["linear_attention"] * 7 + ["full_attention"],
        "max_position_embeddings": 262144,
    }
    gmap = {"host-a:gpu0": make_gpu("host-a:gpu0", 22.0, 24.0)}
    default = plan_replica_groups(
        gmap,
        _pin_config(
            max_input_prompt_length="auto",
            max_num_seqs=1,
            max_output_prompt_length=256,
            attention_cls=None,
        ),
        hf,
        {},
        model_id="hybrid-default",
    )
    raised = plan_replica_groups(
        gmap,
        _pin_config(
            max_input_prompt_length="auto",
            max_num_seqs=1,
            max_output_prompt_length=256,
            attention_cls=None,
            auto_hybrid_input_cap=65536,
        ),
        hf,
        {},
        model_id="hybrid-raised",
    )
    d_in = next(iter(default.values()))["plan"]["max_input_prompt_length"]
    r_in = next(iter(raised.values()))["plan"]["max_input_prompt_length"]
    assert d_in <= AUTO_HYBRID_INPUT_CAP
    assert r_in > d_in
    assert r_in <= 65536
