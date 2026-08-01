"""B — planner / VRAM packing."""
import pytest

from ray_hive.core.model_specs.planner import (
    build_vram_reqs,
    effective_input_len,
    is_pooling_vram,
    normalize_hf_config,
    plan_deployment,
)
from ray_hive.core.ray_utils.placement import fixed_non_kv_gb, plan_lengths
from ray_hive.errors import ModelDoesNotFitError


def test_normalize_flattens_text_config(tiny_hf_dense):
    nested = {
        "model_type": "gemma",
        "text_config": tiny_hf_dense,
        "vision_config": {"hidden_size": 16},
    }
    flat = normalize_hf_config(nested)
    assert flat["hidden_size"] == tiny_hf_dense["hidden_size"]
    assert flat["vision_config"]["hidden_size"] == 16


def test_plan_deployment_basic(tiny_hf_dense):
    vr = build_vram_reqs(tiny_hf_dense)
    plan = plan_deployment(
        vr,
        vram_budget_gb=10.0,
        live_total_vram_gb=24.0,
        max_model_len=512,
        input_len=256,
        output_len=256,
    )
    assert plan["max_num_seqs"] >= 1
    assert plan["max_num_batched_tokens"] >= 1
    assert 0 < plan["gpu_memory_utilization"] <= 1.0


def test_sleep_mode_increases_fixed_non_kv(tiny_hf_dense):
    vr = build_vram_reqs(tiny_hf_dense)
    base = fixed_non_kv_gb(vr, sleep_mode=False)
    sleep = fixed_non_kv_gb(vr, sleep_mode=True)
    assert sleep > base


def test_tp_divides_weights(tiny_hf_dense):
    vr1 = build_vram_reqs(tiny_hf_dense, tensor_parallel_size=1)
    vr2 = build_vram_reqs(tiny_hf_dense, tensor_parallel_size=2)
    assert vr2.calc_weights_gb() == pytest.approx(vr1.calc_weights_gb() / 2)


def test_pooling_plan_lengths(tiny_hf_dense):
    vr = build_vram_reqs(tiny_hf_dense, runner="pooling")
    assert is_pooling_vram(vr)
    cfg = {
        "max_input_prompt_length": 128,
        "max_output_prompt_length": 64,
    }
    inp, out, mml, pooling = plan_lengths(vr, cfg)
    assert pooling and out == 0 and mml == inp


def test_overrides(tiny_hf_dense):
    vr = build_vram_reqs(tiny_hf_dense)
    plan = plan_deployment(
        vr,
        vram_budget_gb=12.0,
        live_total_vram_gb=24.0,
        max_model_len=256,
        input_len=128,
        output_len=128,
        max_num_seqs_override=4,
        max_num_batched_tokens_override=256,
    )
    assert plan["max_num_seqs"] == 4
    assert plan["max_num_batched_tokens"] == 256


def test_too_small_budget_raises(tiny_hf_dense):
    vr = build_vram_reqs(tiny_hf_dense)
    with pytest.raises((ModelDoesNotFitError, AssertionError, ValueError)):
        plan_deployment(
            vr,
            vram_budget_gb=0.01,
            live_total_vram_gb=24.0,
            max_model_len=2048,
            input_len=1024,
            output_len=1024,
        )
