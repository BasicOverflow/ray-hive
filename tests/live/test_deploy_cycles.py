"""Live Deploy A–I cycles with dynamic GPU claims (single pytest process)."""
import json
import time
import urllib.request

import pytest

from tests.live.cluster_sched import GpuNeed, run_cycles_parallel
from tests.live.conftest import EMBED_MODEL, MM_MODEL, TEXT_MODEL, VLLM_TEXT, uniq

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


def _hosts(state: dict) -> set[str]:
    return {k.split(":")[0] for k in state}


def _fp8_need(**kwargs) -> GpuNeed:
    kwargs.setdefault("min_compute_cap", FP8_CAP)
    return GpuNeed(**kwargs)


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

    # D — topology across hosts
    if len(_hosts(state)) >= 2:
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

        cycles.append((
            _fp8_need(min_free_gb=4.0, count=2, distinct_hosts=True, name="D"),
            cycle_d,
        ))

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

    # F — short sleep/idle (HTTP)
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
    hard, timeouts, ok = [], [], 0
    for (need, _), err in zip(cycles, errs):
        if err is None:
            ok += 1
        elif isinstance(err, TimeoutError):
            timeouts.append(f"{need.name}: {err}")
        else:
            hard.append(f"{need.name}: {err}")
    if hard:
        pytest.fail("live cycle failures: " + "; ".join(hard))
    if ok == 0:
        pytest.skip("no free GPUs for any cycle: " + "; ".join(timeouts))


@pytest.mark.nightly
def test_live_resilience_registry_kill(hive, scheduler):
    """I — kill registry singleton and redeploy (nightly / exclusive)."""
    from ray_hive.core.ray_utils.lifecycle import kill_gpu_registry

    claim = None
    mid = uniq("i")
    try:
        claim = scheduler.claim(_fp8_need(min_free_gb=4.0, count=1, name="I"))
        hive.deploy_model(
            model_id=mid, model_name=TEXT_MODEL,
            max_input_prompt_length=256, max_output_prompt_length=64,
            replicas=1, gpu=claim.gpu_keys[0], vllm_kwargs=VLLM_TEXT,
        )
        hive.shutdown(mid)
        kill_gpu_registry()
        mid2 = uniq("i2")
        hive.deploy_model(
            model_id=mid2, model_name=TEXT_MODEL,
            max_input_prompt_length=256, max_output_prompt_length=64,
            replicas=1, gpu=claim.gpu_keys[0], vllm_kwargs=VLLM_TEXT,
        )
        hive.shutdown(mid2)
    except TimeoutError as e:
        pytest.skip(str(e))
    finally:
        if claim is not None:
            scheduler.release(claim)
