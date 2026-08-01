"""I — pooling / embeddings mode planning."""
from ray_hive.core.model_specs.factory import is_pooling_kwargs
from ray_hive.core.model_specs.planner import build_vram_reqs, is_pooling_vram
from ray_hive.core.ray_utils.placement import plan_lengths
from ray_hive.errors import UnsupportedModeError, http_status_for


def test_pooling_kwargs():
    assert is_pooling_kwargs({"runner": "pooling"})
    assert is_pooling_kwargs({"task": "embed"})
    assert not is_pooling_kwargs({})


def test_pooling_vram_flag(tiny_hf_dense):
    vr = build_vram_reqs(tiny_hf_dense, runner="pooling")
    assert is_pooling_vram(vr)
    inp, out, mml, pooling = plan_lengths(vr, {
        "max_input_prompt_length": 128,
        "max_output_prompt_length": 64,
    })
    assert pooling and out == 0 and mml == inp == 128


def test_unsupported_mode_http():
    assert http_status_for(UnsupportedModeError("x")) == 400
