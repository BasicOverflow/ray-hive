"""Shared test doubles and GPU map builders (no Ray / Hub)."""
from ray_hive.core.gpu_alloc import BaseGpuAllocator


def make_gpu(
    key: str,
    available: float,
    total: float | None = None,
    *,
    sm: int = 80,
    tdp: float = 300.0,
    cap: tuple[int, int] = (8, 9),
    bandwidth: float = 1000.0,
    pending: dict | None = None,
    active: dict | None = None,
    name: str = "Test GPU",
) -> dict:
    """Build a registry-shaped GPU view for unit tests."""
    total = float(total if total is not None else max(available, 1.0))
    return {
        "available": float(available),
        "free": float(available),
        "total": total,
        "pending": dict(pending or {}),
        "active": dict(active or {}),
        "specs": {
            "name": name,
            "multiprocessor_count": sm,
            "compute_capability_major": cap[0],
            "compute_capability_minor": cap[1],
            "memory_clock_rate_khz": int(bandwidth),
            # Ray hardware.mem_bandwidth reads these PyCUDA-style fields
            "global_memory_bus_width": 256,
            "memory_clock_rate": float(bandwidth),
            "tdp": tdp,
        },
    }


class FakeAllocator(BaseGpuAllocator):
    """Concrete allocator reading hardware fields from gpu['specs']."""

    def sm_count(self, gpu: dict) -> int:
        return int(gpu["specs"]["multiprocessor_count"])


    def compute_cap(self, gpu: dict) -> tuple[int, int]:
        s = gpu["specs"]
        return (int(s["compute_capability_major"]), int(s["compute_capability_minor"]))


    def approx_tdp(self, gpu: dict) -> float:
        return float(gpu["specs"].get("tdp", 300.0))


    def mem_bandwidth(self, gpu: dict) -> float:
        return float(gpu["specs"].get("memory_clock_rate_khz", 1000.0))


class FakePerformanceAllocator(FakeAllocator):
    def score(self, gpu_key, gpu, hf_params, vllm_kwargs):
        return float(self.sm_count(gpu)) + 1e-6 * self.mem_bandwidth(gpu)


class FakeConserveTdpAllocator(FakeAllocator):
    def score(self, gpu_key, gpu, hf_params, vllm_kwargs):
        tdp = max(self.approx_tdp(gpu), 1.0)
        return -tdp + 1e-3 * self.sm_count(gpu)


TINY_HF_DENSE = {
    "model_type": "qwen2",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 128,
    "vocab_size": 256,
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": True,
}

TINY_HF_FP8 = {
    **TINY_HF_DENSE,
    "torch_dtype": "float8",
    "quantization_config": {"quant_method": "fp8"},
}

TINY_HF_MM = {
    **TINY_HF_DENSE,
    "vision_config": {"hidden_size": 32, "num_hidden_layers": 1},
    "audio_config": {"hidden_size": 32},
}
