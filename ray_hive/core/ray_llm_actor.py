"""
Ray Serve vLLM replica — one LLM engine pinned to a single GPU.
"""
import os

from ray import serve
from vllm import LLM


@serve.deployment(
    ray_actor_options={"num_gpus": 0.01, "memory": 2 * 1024 * 1024 * 1024},
    autoscaling_config=None,
    num_replicas=1,
)
class RayLLMActor(LLM):
    """Ray Serve replica — vLLM LLM engine pinned to one GPU."""

    def __init__(self, model_id: str, target_gpu_id: str, engine_kwargs: dict):
        """Pin to target GPU and initialize vLLM engine with pre-computed settings."""
        os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu_id
        os.environ["VLLM_DISABLE_MARLIN"] = "1"
        self.model_id = model_id
        super().__init__(**engine_kwargs)


    def generate(self, prompts, sampling_params=None):
        """Serve-callable generate wrapper."""
        return super().generate(prompts, sampling_params=sampling_params)


    def chat(self, messages, sampling_params=None):
        """Serve-callable chat wrapper."""
        return super().chat(messages, sampling_params=sampling_params)
