"""Live Deploy cycles with dynamic GPU claims (single pytest process).

Parallel suite covers A–H. Dedicated tests: J (auto input), K (sleep VRAM hold),
I (nightly registry kill).
"""
import json
import time
import urllib.request

import pytest
import ray

from tests.live.cluster_sched import GpuNeed, UnsatisfiableError, run_cycles_parallel
from tests.live.conftest import (
    EMBED_MODEL,
    MM_MODEL,
    TEXT_MODEL,
    VLLM_TEXT,
    note_vram_skip,
    uniq,
)
from tests.live.vram_gate import wait_for_available

pytestmark = pytest.mark.live

# Qwen3-0.6B-FP8 needs Ada+ (compute capability ≥ 8.9)
FP8_CAP = (8, 9)


def _openai_chat(model_id: str, prompt: str, max_tokens: int = 16) -> str:
    from ray_hive.core.ray_utils import serve_base_url

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{serve_base_url()}/{model_id}/v1/chat/completions",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def _fp8_need(**kwargs) -> GpuNeed:
    kwargs.setdefault("min_compute_cap", FP8_CAP)
    return GpuNeed(**kwargs)


def _wait_replica_pending(hive, pin: str, replica_ids: list[str], *, max_wait_s: float = 45.0):
    """Poll registry until a replica id moves active → pending (sleep hold)."""
    deadline = time.time() + max_wait_s
    last = None
    while time.time() < deadline:
        view = (hive.get_vram_state() or {}).get(pin) or {}
        last = view
        pending = view.get("pending") or {}
        if any(rid in pending for rid in replica_ids):
            return view
        time.sleep(1.0)
    raise TimeoutError(
        f"replicas {replica_ids} never entered pending on {pin} within {max_wait_s}s; "
        f"last_view={last}"
    )


def test_live_cycles_parallel(hive, scheduler):
    """Run A–H overlapping when the scheduler can grant disjoint GPUs."""
    state = hive.get_vram_state()
    if not state:
        pytest.skip("empty VRAM registry")

    cycles = []

    # A — two text replicas on distinct GPUs
    def cycle_a(claim):
        mid = uniq("a")
        try:
            status = hive.deploy_model(
                model_id=mid,
                model_name=TEXT_MODEL,
                max_input_prompt_length=256,
                max_output_prompt_length=64,
                replicas=min(2, len(claim.gpu_keys)),
                gpu=claim.gpu_keys[:2] if len(claim.gpu_keys) >= 2 else claim.gpu_keys[0],
                vllm_kwargs=VLLM_TEXT,
            )
            assert status["status"] == "ready"
            assert _openai_chat(mid, "Say hi.")
        finally:
            hive.shutdown(mid)

    cycles.append((_fp8_need(min_free_gb=4.0, count=2, name="A"), cycle_a))

    # B — policy smoke (1 GPU)
    def cycle_b(claim):
        from ray_hive.core.ray_gpu_alloc import RayConserveTdpAllocator

        mid = uniq("b")
        try:
            status = hive.deploy_model(
                model_id=mid,
                model_name=TEXT_MODEL,
                max_input_prompt_length=256,
                max_output_prompt_length=64,
                replicas=1,
                gpu=claim.gpu_keys[0],
                allocation_cls=RayConserveTdpAllocator,
                vllm_kwargs=VLLM_TEXT,
            )
            assert status["status"] == "ready"
        finally:
            hive.shutdown(mid)

    cycles.append((_fp8_need(min_free_gb=4.0, count=1, name="B"), cycle_b))

    # C — intentional share: two models on one GPU (exclusive claim for duration)
    def cycle_c(claim):
        m1, m2 = uniq("c1"), uniq("c2")
        pin = claim.gpu_keys[0]
        try:
            hive.deploy_model(
                model_id=m1, model_name=TEXT_MODEL,
                max_input_prompt_length=128, max_output_prompt_length=32,
                replicas=1, gpu=pin, vllm_kwargs={**VLLM_TEXT, "max_num_seqs": 2},
            )
            hive.deploy_model(
                model_id=m2, model_name=TEXT_MODEL,
                max_input_prompt_length=128, max_output_prompt_length=32,
                replicas=1, gpu=pin, vllm_kwargs={**VLLM_TEXT, "max_num_seqs": 2},
            )
            assert _openai_chat(m1, "A") and _openai_chat(m2, "B")
        finally:
            hive.shutdown(m1)
            hive.shutdown(m2)

    cycles.append((_fp8_need(min_free_gb=8.0, count=1, name="C"), cycle_c))

    # D — topology across hosts (needs ≥2 Ada hosts for FP8 text model)
    need_d = _fp8_need(min_free_gb=4.0, count=2, distinct_hosts=True, name="D")
    if scheduler.structurally_possible(need_d, state):
        def cycle_d(claim):
            mid = uniq("d")
            try:
                status = hive.deploy_model(
                    model_id=mid, model_name=TEXT_MODEL,
                    max_input_prompt_length=256, max_output_prompt_length=64,
                    replicas=2, gpu=claim.gpu_keys,
                    vllm_kwargs=VLLM_TEXT,
                )
                assert status["status"] == "ready"
            finally:
                hive.shutdown(mid)

        cycles.append((need_d, cycle_d))
    else:
        note_vram_skip(
            "D: skipped — cluster has no 2-host Ada (cap>=8.9) topology for FP8"
        )

    # E — same-host TP=2
    def cycle_e(claim):
        mid = uniq("e")
        try:
            status = hive.deploy_model(
                model_id=mid, model_name=TEXT_MODEL,
                max_input_prompt_length=256, max_output_prompt_length=64,
                replicas=1, gpu=claim.gpu_keys,
                vllm_kwargs=VLLM_TEXT,
            )
            assert status["status"] == "ready"
            assert _openai_chat(mid, "TP ok")
        finally:
            hive.shutdown(mid)

    cycles.append((_fp8_need(min_free_gb=3.0, count=2, same_host=True, name="E"), cycle_e))

    # F — short sleep/idle (HTTP wake)
    def cycle_f(claim):
        mid = uniq("f")
        try:
            hive.deploy_model(
                model_id=mid, model_name=TEXT_MODEL,
                max_input_prompt_length=256, max_output_prompt_length=64,
                replicas=1, gpu=claim.gpu_keys[0],
                sleep_timeout=8, idle_timeout=25,
                vllm_kwargs=VLLM_TEXT,
            )
            assert _openai_chat(mid, "hot")
            time.sleep(12)
            assert _openai_chat(mid, "wake")
        finally:
            hive.shutdown(mid)

    cycles.append((_fp8_need(min_free_gb=4.0, count=1, name="F"), cycle_f))

    # G — embeddings (no FP8 arch requirement)
    def cycle_g(claim):
        mid = uniq("g")
        try:
            status = hive.deploy_model(
                model_id=mid, model_name=EMBED_MODEL,
                max_input_prompt_length=128, max_output_prompt_length=0,
                replicas=1, gpu=claim.gpu_keys[0],
                vllm_kwargs={"runner": "pooling", "trust_remote_code": True},
            )
            assert status["status"] == "ready"
            from ray_hive.inference import inference
            vecs = inference("hello world", model_id=mid)
            assert vecs
        finally:
            hive.shutdown(mid)

    cycles.append((GpuNeed(min_free_gb=2.0, count=1, name="G"), cycle_g))

    # H — MM vision (heavier; may TimeoutError → skip)
    def cycle_h(claim):
        mid = uniq("h")
        try:
            status = hive.deploy_model(
                model_id=mid, model_name=MM_MODEL,
                max_input_prompt_length=512, max_output_prompt_length=64,
                replicas=1, gpu=claim.gpu_keys[0],
                vllm_kwargs={
                    "trust_remote_code": True,
                    "limit_mm_per_prompt": {"image": 1},
                    "max_num_seqs": 2,
                },
            )
            assert status["status"] == "ready"
        finally:
            hive.shutdown(mid)

    cycles.append((GpuNeed(min_free_gb=10.0, count=1, name="H"), cycle_h))

    errs = run_cycles_parallel(scheduler, cycles, max_workers=min(2, len(cycles)))
    hard, soft, ok = [], [], 0
    for (need, _), err in zip(cycles, errs):
        if err is None:
            ok += 1
        elif isinstance(err, (TimeoutError, UnsatisfiableError)):
            msg = f"{need.name}: {err}"
            soft.append(msg)
            note_vram_skip(msg)
        else:
            hard.append(f"{need.name}: {err}")
    if hard:
        pytest.fail("live cycle failures: " + "; ".join(hard))
    if ok == 0:
        pytest.skip("no free GPUs for any cycle: " + "; ".join(soft))


def test_live_auto_input_deploys(hive, scheduler):
    """J — deploy with max_input_prompt_length=\"auto\" and run chat."""
    claim = None
    mid = uniq("j")
    try:
        claim = scheduler.claim(_fp8_need(min_free_gb=4.0, count=1, name="J"))
        pin = claim.gpu_keys[0]
        kwargs = {**VLLM_TEXT, "max_num_seqs": 4}

        planned = hive.estimate_vram(
            TEXT_MODEL,
            max_input_prompt_length="auto",
            max_output_prompt_length=64,
            replicas=1,
            gpu=pin,
            vllm_kwargs=kwargs,
        )
        est = next(iter(planned.values()))
        est_plan = est["plan"]
        assert est_plan["max_input_prompt_length_auto"] is True
        assert est_plan["max_input_prompt_length"] >= 256
        assert est_plan["max_num_seqs"] == 4
        assert est["max_model_len"] == est_plan["max_input_prompt_length"] + 64

        status = hive.deploy_model(
            model_id=mid,
            model_name=TEXT_MODEL,
            max_input_prompt_length="auto",
            max_output_prompt_length=64,
            replicas=1,
            gpu=pin,
            vllm_kwargs=kwargs,
        )
        assert status["status"] == "ready"
        dep_plan = next(iter(status["replicas"].values()))["plan"]
        assert dep_plan["max_input_prompt_length_auto"] is True
        assert dep_plan["max_input_prompt_length"] >= 256
        assert dep_plan["max_num_seqs"] == 4
        assert dep_plan["max_input_prompt_length"] == est_plan["max_input_prompt_length"]
        assert _openai_chat(mid, "Say ok in one word.")
    except TimeoutError as e:
        note_vram_skip(f"J auto-input: {e}")
        pytest.skip(str(e))
    finally:
        hive.shutdown(mid)
        if claim is not None:
            scheduler.release(claim)


def test_live_sleep_vram_hold(hive, scheduler):
    """K — after sleep, planned VRAM stays pending so freed smi cannot be stolen."""
    from ray_hive.core.gpu_registry import get_gpu_registry

    claim = None
    mid = uniq("k")
    probe_id = f"steal-probe-{uniq('k')}"
    registry = get_gpu_registry()
    sleep_s = 8
    try:
        claim = scheduler.claim(_fp8_need(min_free_gb=4.0, count=1, name="K"))
        pin = claim.gpu_keys[0]
        status = hive.deploy_model(
            model_id=mid,
            model_name=TEXT_MODEL,
            max_input_prompt_length=256,
            max_output_prompt_length=64,
            replicas=1,
            gpu=pin,
            sleep_timeout=sleep_s,
            idle_timeout=sleep_s + 90,
            vllm_kwargs={**VLLM_TEXT, "max_num_seqs": 8},
        )
        assert status["status"] == "ready"
        replica_ids = list(status["replicas"].keys())
        reserved = float(next(iter(status["replicas"].values()))["plan"]["total_vram_gb"])
        assert _openai_chat(mid, "hot")

        view = _wait_replica_pending(hive, pin, replica_ids, max_wait_s=sleep_s + 35)
        pending = view.get("pending") or {}
        active = view.get("active") or {}
        assert any(rid in pending for rid in replica_ids)
        assert not any(rid in active for rid in replica_ids)
        pending_sum = float(sum(pending.values()))
        assert pending_sum >= reserved * 0.9
        assert view["available"] == pytest.approx(
            float(view["free"]) - pending_sum, abs=0.25
        )

        gap = float(view["free"]) - float(view["available"])
        assert gap >= reserved * 0.9
        steal_need = float(view["available"]) + max(gap * 0.5, 0.5)
        assert steal_need > float(view["available"])
        stole = ray.get(registry.reserve_replica.remote(probe_id, pin, steal_need))
        assert stole is False, (
            f"sleep hold failed: reserved {steal_need:.2f}GB on {pin} while "
            f"available={view['available']:.2f} free={view['free']:.2f} pending={pending}"
        )

        assert _openai_chat(mid, "wake")
        deadline = time.time() + 60.0
        while time.time() < deadline:
            awake = (hive.get_vram_state() or {}).get(pin) or {}
            if any(rid in (awake.get("active") or {}) for rid in replica_ids):
                break
            time.sleep(1.0)
        else:
            pytest.fail(f"replicas {replica_ids} never returned to active after wake")
    except TimeoutError as e:
        note_vram_skip(f"K sleep-hold: {e}")
        pytest.skip(str(e))
    finally:
        ray.get(registry.clear_replicas.remote([probe_id]))
        hive.shutdown(mid)
        if claim is not None:
            scheduler.release(claim)


@pytest.mark.nightly
def test_live_resilience_registry_kill(hive, scheduler):
    """I — kill registry singleton and redeploy once DaemonSet VRAM is real again."""
    from ray_hive.core.ray_utils.lifecycle import kill_gpu_registry

    claim = None
    mid = uniq("i")
    pin = None
    try:
        claim = scheduler.claim(_fp8_need(min_free_gb=4.0, count=1, name="I"))
        pin = claim.gpu_keys[0]
        hive.deploy_model(
            model_id=mid, model_name=TEXT_MODEL,
            max_input_prompt_length=256, max_output_prompt_length=64,
            replicas=1, gpu=pin, vllm_kwargs=VLLM_TEXT,
        )
        hive.shutdown(mid)
        # Wait for physical free *before* kill — otherwise the new registry is
        # seeded with nvidia-smi free=0 while the engine is still releasing CUDA.
        wait_for_available(hive.get_vram_state, pin, 4.0, max_wait_s=180.0)
        kill_gpu_registry()
        # Fresh actor starts empty / free=0 until DaemonSet reports again.
        wait_for_available(hive.get_vram_state, pin, 4.0, max_wait_s=180.0)
        mid2 = uniq("i2")
        hive.deploy_model(
            model_id=mid2, model_name=TEXT_MODEL,
            max_input_prompt_length=256, max_output_prompt_length=64,
            replicas=1, gpu=pin, vllm_kwargs=VLLM_TEXT,
        )
        hive.shutdown(mid2)
    except TimeoutError as e:
        note_vram_skip(f"I resilience: {e}")
        pytest.skip(str(e))
    finally:
        if claim is not None:
            scheduler.release(claim)
