"""
Ray Serve vLLM replica — one LLM engine pinned to a single GPU.
"""
import asyncio
import os

from ray import serve
from vllm import LLM


@serve.deployment(
    ray_actor_options={"num_gpus": 0.01, "memory": 2 * 1024 * 1024 * 1024},
    autoscaling_config=None,
    num_replicas=1,
    max_ongoing_requests=64,
)
class RayLLMActor(LLM):
    """Ray Serve replica — vLLM LLM engine pinned to one GPU."""

    def __init__(self, model_id: str, target_gpu_id: str, engine_kwargs: dict):
        """Pin to target GPU and initialize vLLM engine with pre-computed settings."""
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu_id
        self.model_id = model_id
        self._infer_lock = None
        super().__init__(**engine_kwargs)


    def _lock(self) -> asyncio.Lock:
        """Lazily bind the infer lock to the replica's running event loop."""
        if self._infer_lock is None:
            self._infer_lock = asyncio.Lock()
        return self._infer_lock


    async def generate(self, prompts, sampling_params=None):
        """Serve-callable generate — offload blocking vLLM onto a worker thread."""
        async with self._lock():
            return await asyncio.to_thread(
                super().generate, prompts, sampling_params=sampling_params
            )


    async def chat(self, messages, sampling_params=None):
        """Serve-callable chat — offload blocking vLLM onto a worker thread."""
        async with self._lock():
            return await asyncio.to_thread(
                super().chat, messages, sampling_params=sampling_params
            )
