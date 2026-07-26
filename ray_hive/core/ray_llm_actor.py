"""
Ray Serve vLLM replica — AsyncLLM engine pinned to one or more GPUs (same-node TP).
"""
import asyncio
import os
import uuid

from ray import serve
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM


@serve.deployment(
    ray_actor_options={"num_gpus": 0},
    autoscaling_config=None,
    num_replicas=1,
    max_ongoing_requests=64,
)
class RayLLMActor:
    """Ray Serve replica — vLLM AsyncLLM on one GPU or a same-node TP group."""

    async def __init__(self, model_id: str, target_gpu_id: str, engine_kwargs: dict):
        """
        Pin to target GPU id(s) and initialize AsyncLLM.

        target_gpu_id is a single local id ("0") or comma-separated ids ("0,1")
        for tensor_parallel_size > 1. Serve still exposes one handle per replica.
        """
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu_id
        if "," in target_gpu_id:
            os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
        engine_kwargs = dict(engine_kwargs)
        ktc = engine_kwargs.get("kv_transfer_config")
        if ktc is not None or engine_kwargs.get("kv_offloading_size"):
            os.environ["VLLM_USE_SIMPLE_KV_OFFLOAD"] = "1"
        if isinstance(ktc, dict):
            from vllm.config import KVTransferConfig
            engine_kwargs["kv_transfer_config"] = KVTransferConfig(**ktc)
        self.model_id = model_id
        # Build on Serve's running loop so AsyncLLM's output_handler attaches correctly.
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs))


    async def sleep(self, level: int = 1):
        """Hibernate engine (level 1: weights → CPU, discard KV)."""
        await self.engine.sleep(level=level)


    async def wake_up(self):
        """Restore engine from sleep."""
        await self.engine.wake_up()


    def _params(self, sampling_params, kind: RequestOutputKind) -> SamplingParams:
        """Clone (or default) sampling params with the given output_kind."""
        if sampling_params is None:
            return SamplingParams(output_kind=kind)
        params = sampling_params.clone()
        params.output_kind = kind
        return params


    async def _generate_one(self, prompt: str, sampling_params: SamplingParams):
        """Run one prompt to completion; return the final RequestOutput."""
        final = None
        async for output in self.engine.generate(
            prompt,
            sampling_params,
            request_id=uuid.uuid4().hex,
        ):
            final = output
            if output.finished:
                break
        return final


    async def generate(self, prompts, sampling_params=None):
        """Full-result generate (FINAL_ONLY) — continuous-batches concurrent prompts."""
        if isinstance(prompts, str):
            prompts = [prompts]
        params = self._params(sampling_params, RequestOutputKind.FINAL_ONLY)
        return list(await asyncio.gather(*[
            self._generate_one(prompt, params) for prompt in prompts
        ]))


    async def chat(self, messages, sampling_params=None):
        """Full-result chat via tokenizer chat template + generate."""
        if messages and isinstance(messages[0], dict):
            conversations = [messages]
        else:
            conversations = list(messages)
        tokenizer = self.engine.get_tokenizer()
        prompts = [
            tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conv in conversations
        ]
        return await self.generate(prompts, sampling_params)


    async def generate_stream(self, prompt: str, sampling_params=None):
        """Yield text deltas (DELTA) for a single prompt until finished."""
        params = self._params(sampling_params, RequestOutputKind.DELTA)
        async for output in self.engine.generate(
            prompt,
            params,
            request_id=uuid.uuid4().hex,
        ):
            if output.outputs:
                text = output.outputs[0].text
                if text:
                    yield text
            if output.finished:
                break
