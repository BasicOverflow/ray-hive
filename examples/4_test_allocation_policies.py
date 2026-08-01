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
    RayPerformanceAllocator,
)
from ray_hive.inference import inference_batch
from ray_hive.core.ray_utils import approx_tdp, compute_cap, info, sm_count, success

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL_NAME = "Qwen/Qwen3-0.6B-FP8"
REPLICAS = 2
MAX_IN, MAX_OUT = 1024, 2048
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
]

scheduler = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=True)
gpu_map = scheduler.get_vram_state()

VLLM_KWARGS = dict(
    max_num_seqs=32,
    trust_remote_code=True,
    reasoning_parser="qwen3",
    default_chat_template_kwargs={"enable_thinking": False},
)

scheduler.estimate_vram(
    MODEL_NAME,
    max_input_prompt_length=MAX_IN,
    max_output_prompt_length=MAX_OUT,
    vllm_kwargs=VLLM_KWARGS,
)

info("Cluster GPUs (allocation inputs):")
for gpu_key, gpu in sorted(gpu_map.items()):
    info(
        f"  {gpu_key}: avail={gpu.get('available', 0):.1f}GB  "
        f"sm={sm_count(gpu)}  cap={compute_cap(gpu)}  "
        f"tdp~{approx_tdp(gpu):.0f}W  name={gpu.get('specs', {}).get('name', '?')}"
    )

# FP8 SM89+ taint applies to every policy when HF/vLLM signals FP8.
fp8_hf = {"quantization_config": {"quant_method": "fp8"}}
info(f"Select preview (replicas={REPLICAS}, FP8 HF signal):")
for policy in policies:
    allocator = policy["allocation_cls"]()
    chosen = allocator.select(gpu_map, REPLICAS, 0.5, fp8_hf, {})
    info(f"  {policy['allocation_cls'].__name__}: {[k for k, _ in chosen]}")

info("Note: RayTensorParallelAllocator is covered in examples/5_tensor_parallel.py.")

for idx, policy in enumerate(policies):
    model_id = policy["model_id"]
    info(f"{policy['description']} ({model_id})")
    status = scheduler.deploy_model(
        model_id=model_id,
        model_name=MODEL_NAME,
        max_input_prompt_length=MAX_IN,
        max_output_prompt_length=MAX_OUT,
        replicas=REPLICAS,
        allocation_cls=policy["allocation_cls"],
        vllm_kwargs=VLLM_KWARGS,
    )
    info(status)

    # Qwen3 non-thinking sampling (model card / deploy docs)
    sample_kwargs = dict(max_tokens=100, temperature=0.0, top_p=0.8, top_k=20)
    _ = inference_batch(PROMPTS[:10], model_id=model_id, **sample_kwargs)

    start = time.time()
    results = inference_batch(PROMPTS, model_id=model_id, **sample_kwargs)
    elapsed = time.time() - start
    success(f"Processed {len(results)} prompts in {elapsed:.3f}s ({len(results) / elapsed:.2f} req/s)")

    scheduler.shutdown(model_id)
    if idx < len(policies) - 1:
        time.sleep(3)
