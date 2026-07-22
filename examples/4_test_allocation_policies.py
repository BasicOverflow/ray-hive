"""Deploy once per implemented allocation policy and print which GPUs each picks."""
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.core.ray_gpu_alloc import (
    RayConserveTdpAllocator,
    RayFp8Allocator,
    RayPerformanceAllocator,
)
from ray_hive.inference import inference_batch
from ray_hive.core.ray_utils import approx_tdp, compute_cap, sm_count

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_NAME = "Qwen/Qwen3-0.6B-FP8"
REPLICAS = 2
PROMPTS = [f"Write a short poem about beer {i}" for i in range(50)]

policies = [
    {
        "model_id": "alloc-performance",
        "description": "RayPerformanceAllocator (top SM / bandwidth)",
        "allocation_cls": RayPerformanceAllocator,
    },
    {
        "model_id": "alloc-conserve-tdp",
        "description": "RayConserveTdpAllocator (prefer lower TDP)",
        "allocation_cls": RayConserveTdpAllocator,
    },
    {
        "model_id": "alloc-fp8",
        "description": "RayFp8Allocator (prefer Ada for FP8 checkpoint)",
        "allocation_cls": RayFp8Allocator,
    },
]

scheduler = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=True)
gpu_map = scheduler.get_vram_state()

print("Cluster GPUs (allocation inputs):")
for gpu_key, gpu in sorted(gpu_map.items()):
    print(
        f"  {gpu_key}: avail={gpu.get('available', 0):.1f}GB  "
        f"sm={sm_count(gpu)}  cap={compute_cap(gpu)}  "
        f"tdp~{approx_tdp(gpu):.0f}W  name={gpu.get('specs', {}).get('name', '?')}"
    )

# HF FP8 checkpoints set quantization_config.quant_method=fp8 — same signal deploy uses.
fp8_hf = {"quantization_config": {"quant_method": "fp8"}}
print(f"\nSelect preview (replicas={REPLICAS}):")
for policy in policies:
    allocator = policy["allocation_cls"]()
    hf = fp8_hf if policy["allocation_cls"] is RayFp8Allocator else {}
    chosen = allocator.select(gpu_map, REPLICAS, 0.5, hf, {})
    print(f"  {policy['allocation_cls'].__name__}: {[k for k, _ in chosen]}")

print("\nNote: RayTensorParallelAllocator is covered in examples/5_tensor_parallel.py.")

for idx, policy in enumerate(policies):
    model_id = policy["model_id"]
    print(f"\n=== {policy['description']} ({model_id}) ===")
    scheduler.deploy_model(
        model_id=model_id,
        model_name=MODEL_NAME,
        max_input_prompt_length=1024,
        max_output_prompt_length=2048,
        replicas=REPLICAS,
        allocation_cls=policy["allocation_cls"],
        # HF model card / Qwen vLLM docs (enable-reasoning is deprecated; qwen3 since 0.9)
        trust_remote_code=True,
        reasoning_parser="qwen3",
        default_chat_template_kwargs={"enable_thinking": False},
    )
    time.sleep(2)

    # Qwen3 non-thinking sampling (model card / deploy docs)
    sample_kwargs = dict(max_tokens=100, temperature=0.0, top_p=0.8, top_k=20)
    _ = inference_batch(PROMPTS[:10], model_id=model_id, **sample_kwargs)
    time.sleep(2)

    start = time.time()
    results = inference_batch(PROMPTS, model_id=model_id, **sample_kwargs)
    elapsed = time.time() - start
    print(f"Processed {len(results)} prompts in {elapsed:.3f}s ({len(results) / elapsed:.2f} req/s)")

    scheduler.shutdown(model_id)
    if idx < len(policies) - 1:
        time.sleep(3)
