"""Deploy a model via Ray Job (avoids Windows Ray Client InProgressSentinel)."""
import json
import os

MODEL_ID = os.environ.get("NINI_MODEL_ID", "deepseek-ocr2")
MODEL_NAME = os.environ.get("NINI_MODEL_NAME", "deepseek-ai/DeepSeek-OCR-2")
MAX_IN = int(os.environ.get("NINI_MAX_IN", "4096"))
MAX_OUT = int(os.environ.get("NINI_MAX_OUT", "2048"))
# power-save = RayConserveTdpAllocator; default performance otherwise
ALLOCATOR = os.environ.get("NINI_ALLOCATOR", "conserve").strip().lower()


def _vllm_kwargs(model_id: str, model_name: str) -> dict:
    env_seqs = os.environ.get("NINI_MAX_NUM_SEQS")
    # Cap concurrency for VRAM; default 4 for small VL, 1 for 7B+.
    default_seqs = 4
    mid = model_id.lower()
    if any(x in mid for x in ("7b", "8b", "14b", "internvl")):
        default_seqs = 1
    if "deepseek" in mid and "ocr" in mid:
        default_seqs = 2
    base = dict(
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        max_num_seqs=int(env_seqs) if env_seqs else default_seqs,
        # Shrink vision processor cache; one image at a time is enough.
        mm_processor_cache_gb=float(os.environ.get("NINI_MM_CACHE_GB", "0")),
    )
    name = f"{model_id} {model_name}".lower()
    # Dedicated OCR models: no prefix cache / mm processor cache.
    if any(s in name for s in ("deepseek", "paddleocr", "lightonocr", "dots", "hunyuanocr", "glm-ocr")):
        base["enable_prefix_caching"] = False
        base["mm_processor_cache_gb"] = 0.0
    if "deepseek" in name and "ocr" in name:
        base.update(
            logits_processors=[
                "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor",
            ],
        )
        if not env_seqs:
            base["max_num_seqs"] = 2
    quant = os.environ.get("NINI_QUANTIZATION", "").strip()
    if quant:
        base["quantization"] = quant
    elif "awq" in name:
        base["quantization"] = "awq"
    elif "gptq" in name:
        base["quantization"] = "gptq"
    # FP8 KV needs Ada+ (cc>=8.9); Ampere 3090 will be filtered if set — only when asked.
    kv = os.environ.get("NINI_KV_CACHE_DTYPE", "").strip()
    if kv:
        base["kv_cache_dtype"] = kv
    extra = os.environ.get("NINI_VLLM_EXTRA_JSON", "").strip()
    if extra:
        base.update(json.loads(extra))
    return base


def _allocation_cls():
    from ray_hive.core.ray_gpu_alloc import (
        RayConserveTdpAllocator,
        RayPerformanceAllocator,
    )

    if ALLOCATOR in ("conserve", "conserve_tdp", "power", "power_save", "tdp"):
        return RayConserveTdpAllocator
    return RayPerformanceAllocator


def main() -> None:
    import ray
    from ray_hive import RayHive

    if not ray.is_initialized():
        ray.init(address="auto", namespace=os.environ.get("RAY_HIVE_NAMESPACE", "ray_hive"))

    hive = RayHive(address="auto", suppress_logging=False, show_banner=False)
    try:
        hive.shutdown(MODEL_ID)
    except Exception:
        pass
    kwargs = _vllm_kwargs(MODEL_ID, MODEL_NAME)
    alloc = _allocation_cls()
    print(f"deploy {MODEL_ID} ({MODEL_NAME}) allocator={alloc.__name__} kwargs={kwargs}")
    status = hive.deploy_model(
        model_id=MODEL_ID,
        model_name=MODEL_NAME,
        max_input_prompt_length=MAX_IN,
        max_output_prompt_length=MAX_OUT,
        replicas=1,
        allocation_cls=alloc,
        vllm_kwargs=kwargs,
    )
    print(status)
    print("DEPLOY_OK")


if __name__ == "__main__":
    main()
