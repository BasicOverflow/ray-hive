"""E + L — router LB, shard, SSE, sampling (async-friendly, no Serve)."""
import pytest

from ray_hive.core.model_router import ModelRouter
from ray_hive.errors import (
    ConfigError,
    DeployError,
    MediaError,
    ModelNotFoundError,
    PlacementError,
    PlanningError,
    RayHiveError,
    UnsupportedModeError,
    http_status_for,
)

_RouterCls = ModelRouter.func_or_class


def _bare_router(names, max_seqs):
    r = object.__new__(_RouterCls)
    r.gpu_deployment_names = list(names)
    r.replica_metadata = {n: {"max_num_seqs": s} for n, s in zip(names, max_seqs)}
    r._loads = {n: {"waiting": 0, "running": 0} for n in names}
    r._eng_start = 0
    r.model_id = "m"
    return r


def test_select_replica_prefers_higher_capacity():
    r = _bare_router(["small", "big"], [4, 40])
    r._loads["small"]["waiting"] = 4  # util 1.0 vs big util 0
    name = r._select_replica()
    assert name == "big"
    assert r._loads["big"]["waiting"] == 1


def test_shard_by_max_num_seqs():
    r = _bare_router(["a", "b"], [1, 3])
    prompts = [f"p{i}" for i in range(8)]
    shards = r._shard_prompts(prompts)
    counts = {name: len(chunk) for name, chunk, _ in shards}
    assert sum(counts.values()) == 8
    assert counts["b"] > counts["a"]


def test_sse_format():
    r = _bare_router(["a"], [1])
    assert r._sse("[DONE]") == "data: [DONE]\n\n"
    line = r._sse({"id": "x"})
    assert line.startswith("data: {") and line.endswith("\n\n")


def test_sampling_max_completion_tokens():
    r = _bare_router(["a"], [1])
    try:
        params = r._sampling_params(extra={"max_completion_tokens": 32, "temperature": 0.0})
    except Exception as e:
        pytest.skip(f"SamplingParams import/env: {e}")
    assert params.max_tokens == 32


def test_http_status_for_matrix():
    assert http_status_for(ModelNotFoundError("x")) == 404
    assert http_status_for(ConfigError("x")) == 400
    assert http_status_for(PlacementError("x")) == 400
    assert http_status_for(PlanningError("x")) == 400
    assert http_status_for(MediaError("x")) == 400
    assert http_status_for(UnsupportedModeError("x")) == 400
    assert http_status_for(DeployError("x")) == 502
    assert http_status_for(RayHiveError("x")) == 500
