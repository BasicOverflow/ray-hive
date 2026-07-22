"""Test same-node TP: pinned TP=3 smoke, then Nemotron Nano 12B @ TP=2 + cpu_ram_per_instance.

Test 1 — EleutherAI/pythia-160m @ TP=3:
  vLLM TP also requires padded vocab_size % tp == 0 (and heads/KV). StableLM-7B
  has vocab 50432 (not divisible by 3) so it cannot run TP=3 despite 48 heads.
  Pythia-160m: 12 heads, vocab pads to 50304 — both ÷3. Small model; TP plumbing smoke.

Test 2 — nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 @ TP=2 with cpu_ram_per_instance=-1.
  Nemotron has 40 heads (not divisible by 3). Pins ergos-02 gpu1+gpu2 (8/12), skips gpu0.
  Host budget = 85% free VRAM: spill weights if needed, leftover → CPU KV.
"""
import os
import sys
import time
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.inference import inference_batch

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

TP3_MODEL = "EleutherAI/pythia-160m"
NEMOTRON_MODEL = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
NEMOTRON_TP = 2
TP_SIZE = 3
MIN_PER_GPU_GB = 1.0
PROMPTS = [f"Write a short poem about beer {i}" for i in range(1000)]


def _same_node_group(
    gpu_map: dict,
    n: int,
    min_per_gpu_gb: float,
    prefer_heterogeneous: bool = False,
    exclude: set[str] | None = None,
) -> list[str] | None:
    """Best same-host group of size n. Default: matched totals; else prefer spread."""
    exclude = exclude or set()
    by_host: dict[str, list[tuple[str, float, float]]] = {}
    for gpu_key, gpu in gpu_map.items():
        if gpu_key in exclude:
            continue
        by_host.setdefault(gpu_key.split(":")[0], []).append(
            (gpu_key, float(gpu.get("available", 0)), float(gpu.get("total", 0)))
        )

    best = None
    best_rank = None
    for gpus in by_host.values():
        fit = [(k, a, t) for k, a, t in gpus if a >= min_per_gpu_gb]
        if len(fit) < n:
            continue
        for combo in combinations(fit, n):
            avails = [a for _, a, _ in combo]
            totals = [t for _, _, t in combo]
            spread = max(totals) - min(totals)
            if prefer_heterogeneous:
                rank = (spread, min(avails))
            else:
                rank = (-spread, min(avails))
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best = sorted(k for k, _, _ in combo)
    return best


def _run_batch(model_id: str):
    sample_kwargs = dict(max_tokens=100, temperature=0.0, top_p=0.8, top_k=20)
    time.sleep(2)
    _ = inference_batch(PROMPTS[:4], model_id=model_id, **sample_kwargs)
    time.sleep(2)
    start = time.time()
    results = inference_batch(PROMPTS, model_id=model_id, **sample_kwargs)
    elapsed = time.time() - start
    print(f"Processed {len(results)} prompts in {elapsed:.3f}s ({len(results) / elapsed:.2f} req/s)")


scheduler = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=True)
gpu_map = scheduler.get_vram_state()

print("Cluster GPUs:")
for gpu_key, gpu in sorted(gpu_map.items()):
    print(f"  {gpu_key}: avail={gpu.get('available', 0):.1f}GB / total={gpu.get('total', 0):.1f}GB")

group = _same_node_group(gpu_map, TP_SIZE, MIN_PER_GPU_GB)
if group is None:
    lines = "\n".join(
        f"  {k}: avail={g.get('available', 0):.1f}/{g.get('total', 0):.1f}GB"
        for k, g in sorted(gpu_map.items())
    )
    raise SystemExit(
        f"No same-host set of {TP_SIZE} GPUs each with >= {MIN_PER_GPU_GB}GB free "
        f"for pinned TP smoke of {TP3_MODEL}.\n"
        f"Current inventory:\n{lines}"
    )

# print(f"\n=== 1) Pinned TP={TP_SIZE} {TP3_MODEL} via gpu={group} ===")
# scheduler.deploy_model(
#     model_id="pythia-tp3-pin",
#     model_name=TP3_MODEL,
#     gpu=group,
#     max_input_prompt_length=512,
#     max_output_prompt_length=512,
#     replicas=1,
# )
# _run_batch("pythia-tp3-pin")
# scheduler.shutdown("pythia-tp3-pin")
# time.sleep(3)

eq_group = _same_node_group(
    gpu_map,
    NEMOTRON_TP,
    MIN_PER_GPU_GB,
    prefer_heterogeneous=True,
    exclude={"ergos-02-nv:gpu0"},
)
if eq_group is None:
    pair = ["ergos-02-nv:gpu1", "ergos-02-nv:gpu2"]
    if all(k in gpu_map for k in pair):
        eq_group = pair
    else:
        raise SystemExit("Need ergos-02 gpu1+gpu2 for Nemotron TP=2 cpu_ram test")
totals = [float(gpu_map[k].get("total", 0)) for k in eq_group]
print(
    f"\n=== 2) Pinned TP={NEMOTRON_TP} {NEMOTRON_MODEL} + cpu_ram_per_instance=-1 "
    f"via gpu={eq_group} (totals={[round(t, 1) for t in totals]}) ==="
)
scheduler.deploy_model(
    model_id="nemotron-tp2-cpu-ram",
    model_name=NEMOTRON_MODEL,
    gpu=eq_group,
    max_input_prompt_length=1024,
    max_output_prompt_length=1024,
    replicas=1,
    cpu_ram_per_instance=-1,
    trust_remote_code=True,
)
_run_batch("nemotron-tp2-cpu-ram")
scheduler.shutdown("nemotron-tp2-cpu-ram")
