"""
Ray Serve vLLM replica — AsyncLLM engine pinned to one or more GPUs (same-node TP).
"""
import asyncio
import os
import uuid
from ray import serve
from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerBase


class LoadStatLogger(StatLoggerBase):
    """Caches engine waiting/running counts for hive router LB."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_index: int = 0,
        load_ref: dict | None = None,
    ):
        self._load = load_ref if load_ref is not None else {"waiting": 0, "running": 0}


    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
        if scheduler_stats is None:
            return
        self._load["waiting"] = scheduler_stats.num_waiting_reqs
        self._load["running"] = scheduler_stats.num_running_reqs


    def log_engine_initialized(self):
        pass


def _normalize_engine_kwargs(engine_kwargs: dict) -> dict:
    """Map hive/user kwargs onto current AsyncEngineArgs (task → runner)."""
    kw = dict(engine_kwargs)
    if kw.get("task") == "embed":
        kw.setdefault("runner", "pooling")
        kw.pop("task", None)
    return kw


@serve.deployment(
    ray_actor_options={"num_gpus": 0},
    autoscaling_config=None,
    num_replicas=1,
    max_ongoing_requests=64,
)
class RayLLMActor:
    """Ray Serve replica — vLLM AsyncLLM on one GPU or a same-node TP group."""

    async def __init__(
        self,
        model_id: str,
        target_gpu_id: str,
        engine_kwargs: dict,
        pooling: bool = False,
        multimodal: bool = False,
    ):
        """
        Pin to target GPU id(s) and initialize AsyncLLM.

        target_gpu_id is a single local id ("0") or comma-separated ids ("0,1")
        for tensor_parallel_size > 1. Serve still exposes one handle per replica.
        """
        from ray_hive.core.model_specs.factory import is_pooling_kwargs

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu_id
        if "," in target_gpu_id:
            os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
        engine_kwargs = _normalize_engine_kwargs(engine_kwargs)
        engine_kwargs.setdefault("disable_log_stats", True)
        self.model_id = model_id
        self.pooling = pooling or is_pooling_kwargs(engine_kwargs)
        self.multimodal = multimodal
        self._load = {"waiting": 0, "running": 0}
        load_ref = self._load

        def load_logger_factory(vllm_config: VllmConfig, engine_index: int = 0):
            return LoadStatLogger(vllm_config, engine_index, load_ref=load_ref)

        # Build on Serve's running loop so AsyncLLM's output_handler attaches correctly.
        self.engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(**engine_kwargs),
            stat_loggers=[load_logger_factory],
        )


    def get_load(self) -> dict:
        """Return cached engine waiting/running queue depths."""
        return dict(self._load)


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


    async def _generate_one(self, prompt, sampling_params: SamplingParams):
        """Run one prompt (str or PromptType dict) to completion."""
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
        if isinstance(prompts, (str, dict)):
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


    async def generate_stream(self, prompt, sampling_params=None):
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


    async def _embed_one(self, prompt):
        """Run one embed/encode request; return embedding vector."""
        from vllm import PoolingParams

        pooling_params = PoolingParams(task="embed", use_activation=True)
        final = None
        async for output in self.engine.encode(
            prompt,
            pooling_params,
            request_id=uuid.uuid4().hex,
        ):
            final = output
            if getattr(output, "finished", True):
                break
        if final is None:
            return []
        # vLLM pooling outputs: .outputs.data or .outputs.embedding
        outs = final.outputs
        if hasattr(outs, "data") and outs.data is not None:
            data = outs.data
        elif hasattr(outs, "embedding") and outs.embedding is not None:
            data = outs.embedding
        else:
            data = outs
        if hasattr(data, "tolist"):
            return data.tolist()
        return list(data)


    async def embed(self, prompts):
        """Return embedding vectors for str or PromptType prompts."""
        if isinstance(prompts, (str, dict)):
            prompts = [prompts]
        return list(await asyncio.gather(*[
            self._embed_one(prompt) for prompt in prompts
        ]))
