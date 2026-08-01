"""Unit-test ClusterScheduler against fake VRAM snapshots (no Ray)."""
import threading

import pytest

from tests.helpers import make_gpu
from tests.live.cluster_sched import ClusterScheduler, GpuNeed, run_cycles_parallel


def test_claim_release_disjoint():
    state = {
        "a:gpu0": make_gpu("a:gpu0", 20, 24),
        "a:gpu1": make_gpu("a:gpu1", 20, 24),
        "b:gpu0": make_gpu("b:gpu0", 8, 24),
    }
    sched = ClusterScheduler(lambda: state, poll_s=0.01, max_wait_s=1)

    c1 = sched.try_claim(GpuNeed(min_free_gb=10, count=1, name="c1"))
    assert c1 is not None and c1.gpu_keys == ["a:gpu0"]
    c2 = sched.try_claim(GpuNeed(min_free_gb=10, count=1, name="c2"))
    assert c2 is not None and c2.gpu_keys == ["a:gpu1"]
    assert sched.try_claim(GpuNeed(min_free_gb=10, count=1, name="c3")) is None
    sched.release(c1)
    c3 = sched.try_claim(GpuNeed(min_free_gb=10, count=1, name="c3"))
    assert c3 is not None and c3.gpu_keys == ["a:gpu0"]


def test_same_host_tp():
    state = {
        "a:gpu0": make_gpu("a:gpu0", 20, 24),
        "a:gpu1": make_gpu("a:gpu1", 20, 24),
        "b:gpu0": make_gpu("b:gpu0", 20, 24),
    }
    sched = ClusterScheduler(lambda: state)
    c = sched.try_claim(GpuNeed(min_free_gb=10, count=2, same_host=True, name="tp"))
    assert c is not None
    assert {k.split(":")[0] for k in c.gpu_keys} == {"a"}


def test_parallel_runner_disjoint():
    state = {
        "a:gpu0": make_gpu("a:gpu0", 20, 24),
        "b:gpu0": make_gpu("b:gpu0", 20, 24),
    }
    sched = ClusterScheduler(lambda: state, poll_s=0.01, max_wait_s=2)
    seen = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def body(claim):
        with lock:
            seen.append(tuple(claim.gpu_keys))
        barrier.wait(timeout=5)

    errs = run_cycles_parallel(
        sched,
        [
            (GpuNeed(min_free_gb=5, count=1, name="a"), body),
            (GpuNeed(min_free_gb=5, count=1, name="b"), body),
        ],
    )
    assert all(e is None for e in errs)
    assert len(seen) == 2
    assert set(seen[0] + seen[1]) == {"a:gpu0", "b:gpu0"}


def test_claim_timeout():
    state = {"a:gpu0": make_gpu("a:gpu0", 1, 24)}
    sched = ClusterScheduler(lambda: state, poll_s=0.01, max_wait_s=0.05)
    with pytest.raises(TimeoutError):
        sched.claim(GpuNeed(min_free_gb=10, count=1, name="big"))


def test_min_compute_cap_filters_ampere():
    state = {
        "ada:gpu0": make_gpu("ada:gpu0", 20, 24, cap=(8, 9)),
        "amp:gpu0": make_gpu("amp:gpu0", 22, 24, cap=(8, 6)),
    }
    sched = ClusterScheduler(lambda: state)
    c = sched.try_claim(GpuNeed(min_free_gb=4, count=1, min_compute_cap=(8, 9), name="fp8"))
    assert c is not None and c.gpu_keys == ["ada:gpu0"]


def test_distinct_hosts():
    state = {
        "a:gpu0": make_gpu("a:gpu0", 20, 24),
        "a:gpu1": make_gpu("a:gpu1", 19, 24),
        "b:gpu0": make_gpu("b:gpu0", 18, 24),
    }
    sched = ClusterScheduler(lambda: state)
    c = sched.try_claim(GpuNeed(min_free_gb=4, count=2, distinct_hosts=True, name="topo"))
    assert c is not None
    assert {k.split(":")[0] for k in c.gpu_keys} == {"a", "b"}
