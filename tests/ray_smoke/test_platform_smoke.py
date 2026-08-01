"""F — Ray cluster smoke (no LLM deploy)."""
import os

import pytest

pytestmark = pytest.mark.ray


@pytest.fixture(scope="module")
def ray_ok():
    addr = os.environ.get("RAY_ADDRESS")
    if not addr:
        pytest.skip("RAY_ADDRESS not set")
    import ray

    ray.init(address=addr, ignore_reinit_error=True)
    yield ray
    # Leave connection up for other modules; shutdown only if we own it is hard —
    # ray.init with ignore_reinit is fine for smoke.


def test_cluster_resources(ray_ok):
    cluster = ray_ok.cluster_resources()
    assert cluster.get("CPU", 0) > 0


def test_remote_task(ray_ok):
    @ray_ok.remote
    def hello():
        return "ok"

    assert ray_ok.get(hello.remote(), timeout=60) == "ok"


def test_gpu_registry_actor(ray_ok):
    try:
        registry = ray_ok.get_actor("gpu_registry", namespace="system")
    except ValueError:
        pytest.skip("gpu_registry actor not found")
    state = ray_ok.get(registry.get_all_gpus.remote(), timeout=60)
    assert isinstance(state, dict)
