"""
Model router — least-queue load balancing with OpenAI-compatible HTTP ingress.

Exposes /v1/models, /v1/chat/completions, /v1/completions on the router
deployment. Programmatic inference goes through infer() → replica handle.
"""
import asyncio
import time
import uuid
from typing import Literal, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from ray import serve
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

app = FastAPI()

_WARMUP_PROMPTS = 32
_WARMUP_MAX_TOKENS = 32


class TextContentPart(BaseModel):
    """OpenAI chat content part — plain text."""
    type: Literal["text"]
    text: str


class ImageUrlContentPart(BaseModel):
    """OpenAI chat content part — image URL (schema only, not yet supported)."""
    type: Literal["image_url"]
    image_url: dict


class FileContentPart(BaseModel):
    """OpenAI chat content part — file reference (schema only, not yet supported)."""
    type: Literal["file"]
    file: dict


ContentPart = Union[TextContentPart, ImageUrlContentPart, FileContentPart]


class ChatMessage(BaseModel):
    """OpenAI chat message with string or multipart content."""
    role: str
    content: Union[str, list[ContentPart]]


class ChatCompletionRequest(BaseModel):
    """OpenAI /v1/chat/completions request body."""
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False


class CompletionRequest(BaseModel):
    """OpenAI /v1/completions request body."""
    model_config = ConfigDict(extra="allow")
    model: str
    prompt: Union[str, list[str]]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False


@serve.deployment(
    ray_actor_options={"num_cpus": 0.1},
    autoscaling_config=None,
    num_replicas=1,
    max_ongoing_requests=100,
)
@serve.ingress(app)
class ModelRouter:
    """Router with least-queue balancing and OpenAI-compatible HTTP ingress."""

    async def __init__(
        self,
        model_id: str,
        model_name: str,
        gpu_deployment_names: list[str],
        replica_metadata: dict,
        chat_template_kwargs: dict | None = None,
        idle_timeout: int = -1,
    ):
        """Wire replica handles, queue tracking, and tokenizer for token counting."""
        self.model_id = model_id
        self.model_name = model_name
        self.gpu_deployment_names = gpu_deployment_names
        self.replica_metadata = replica_metadata
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.idle_timeout = idle_timeout
        self._handles = None
        self._queue_depth = {name: 0 for name in gpu_deployment_names}
        self._shutting_down = False
        self._last_activity = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        await self._warmup()
        self._last_activity = time.time()
        if self.idle_timeout > 0:
            asyncio.create_task(self._idle_watch())


    def _touch(self):
        """Record inference activity for idle shutdown."""
        self._last_activity = time.time()


    async def _idle_watch(self):
        """Shutdown model after idle_timeout seconds with no inference."""
        from ray_hive.core.deployment import get_deploy_service

        interval = min(5, self.idle_timeout)
        while not self._shutting_down:
            await asyncio.sleep(interval)
            if time.time() - self._last_activity < self.idle_timeout:
                continue
            self._shutting_down = True
            get_deploy_service().shutdown_model.remote(self.model_id)
            return


    async def _warmup(self):
        """Time a fixed batch on each replica and store tokens/sec as throughput."""
        handles = self._get_handles()
        prompts = [f"warmup {i}" for i in range(_WARMUP_PROMPTS)]
        params = SamplingParams(max_tokens=_WARMUP_MAX_TOKENS, temperature=0.0)
        for name in self.gpu_deployment_names:
            start = time.perf_counter()
            outputs = await handles[name].generate.remote(prompts, params)
            elapsed = max(time.perf_counter() - start, 1e-6)
            tokens = sum(
                len(o.prompt_token_ids) + len(o.outputs[0].token_ids)
                for o in outputs
            )
            self.replica_metadata[name]["throughput"] = tokens / elapsed


    def _get_handles(self):
        """Lazily resolve Serve handles for each GPU replica deployment."""
        if self._handles is None:
            self._handles = {
                name: serve.get_deployment_handle(name, app_name=name)
                for name in self.gpu_deployment_names
            }
        return self._handles


    def _to_chat_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert OpenAI chat messages to vLLM chat format."""
        result = []
        for msg in messages:
            if isinstance(msg.content, str):
                result.append({"role": msg.role, "content": msg.content})
                continue
            text_parts = []
            for part in msg.content:
                if part.type != "text":
                    raise HTTPException(status_code=400, detail="multimodal content not supported yet")
                text_parts.append(part.text)
            result.append({"role": msg.role, "content": "\n".join(text_parts)})
        return result


    async def _route_chat(self, messages, max_tokens=None, temperature=None, extra=None):
        """Route a chat conversation to the least-loaded replica."""
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        replica_name = self._select_replica()
        self._queue_depth[replica_name] += 1
        try:
            handle = self._get_handles()[replica_name]
            return await handle.generate.remote(
                [prompt],
                self._sampling_params(max_tokens, temperature, extra),
            )
        finally:
            self._queue_depth[replica_name] = max(0, self._queue_depth[replica_name] - 1)


    def _select_replica(self) -> str:
        """Pick replica with lowest queue_depth / measured tokens/sec."""
        best_name = None
        best_score = float("inf")
        for name in self.gpu_deployment_names:
            throughput = max(self.replica_metadata[name]["throughput"], 1e-6)
            score = self._queue_depth[name] / throughput
            if score < best_score:
                best_score = score
                best_name = name
        return best_name


    def _sampling_params(self, max_tokens=None, temperature=None, extra=None) -> SamplingParams:
        """Build vLLM SamplingParams from request fields."""
        kwargs = dict(extra or {})
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        guided_json = kwargs.pop("guided_json", None)
        if guided_json is not None:
            kwargs["structured_outputs"] = StructuredOutputsParams(json=guided_json)
        return SamplingParams(**kwargs)


    def _extract_texts(self, outputs) -> list[str]:
        """Extract generated text strings from vLLM RequestOutput list."""
        return [o.outputs[0].text for o in outputs]


    def _shard_prompts(self, prompts: list[str]) -> list[tuple[str, list[str], list[int]]]:
        """Split prompts across replicas in proportion to measured tokens/sec."""
        names = self.gpu_deployment_names
        weights = [max(self.replica_metadata[n]["throughput"], 1e-6) for n in names]
        total_w = sum(weights)
        n = len(prompts)

        shards = []
        start = 0
        for i, (name, weight) in enumerate(zip(names, weights)):
            if i == len(names) - 1:
                count = n - start
            else:
                count = int(n * weight / total_w)
            end = start + count
            if count > 0:
                shards.append((name, prompts[start:end], list(range(start, end))))
            start = end
        return shards


    async def _dispatch_prompts(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        """Shard a prompt batch by measured tokens/sec and run replicas concurrently."""
        handles = self._get_handles()
        if len(prompts) == 1:
            name = self._select_replica()
            self._queue_depth[name] += 1
            try:
                outputs = await handles[name].generate.remote(prompts, sampling_params)
                return self._extract_texts(outputs)
            finally:
                self._queue_depth[name] = max(0, self._queue_depth[name] - 1)

        shards = self._shard_prompts(prompts)
        for name, chunk, _ in shards:
            self._queue_depth[name] += len(chunk)
        try:
            outputs = await asyncio.gather(*[
                handles[name].generate.remote(chunk, sampling_params)
                for name, chunk, _ in shards
            ])
            texts = [None] * len(prompts)
            for (_, _, idxs), outs in zip(shards, outputs):
                for local_i, global_i in enumerate(idxs):
                    texts[global_i] = outs[local_i].outputs[0].text
            return texts
        finally:
            for name, chunk, _ in shards:
                self._queue_depth[name] = max(0, self._queue_depth[name] - len(chunk))


    async def _route_text(self, prompt, max_tokens=None, temperature=None, extra=None):
        """Route text prompts across replicas by capacity."""
        prompts = [prompt] if isinstance(prompt, str) else [str(p) for p in prompt]
        return await self._dispatch_prompts(
            prompts,
            self._sampling_params(max_tokens, temperature, extra),
        )


    def _openai_chat_response(self, text: str, prompt: str) -> dict:
        """Build OpenAI chat completion response dict."""
        prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        completion_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "model": self.model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
        }


    def _openai_completion_response(self, text: str, prompt: str) -> dict:
        """Build OpenAI text completion response dict."""
        prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        completion_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "model": self.model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
        }


    @app.get("/v1/models")
    async def list_models(self):
        """OpenAI-compatible model list endpoint."""
        return {"object": "list", "data": [{"id": self.model_id, "object": "model", "owned_by": "ray-hive"}]}


    @app.get("/v1/models/{model_id}")
    async def get_model(self, model_id: str):
        """OpenAI-compatible single model endpoint."""
        return {"id": self.model_id, "object": "model", "owned_by": "ray-hive"}


    @app.post("/v1/chat/completions")
    async def chat_completions(self, request: ChatCompletionRequest):
        """OpenAI-compatible chat completions endpoint (text-only)."""
        self._touch()
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true not supported yet")
        messages = self._to_chat_messages(request.messages)
        extra = request.model_dump(exclude={"model", "messages", "max_tokens", "temperature", "stream"})
        outputs = await self._route_chat(messages, request.max_tokens, request.temperature, extra)
        text = self._extract_texts(outputs)[0] if outputs else ""
        prompt = outputs[0].prompt if outputs else ""
        return self._openai_chat_response(text, prompt)


    @app.post("/v1/completions")
    async def completions(self, request: CompletionRequest):
        """OpenAI-compatible text completions endpoint."""
        self._touch()
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true not supported yet")
        prompt = request.prompt if isinstance(request.prompt, str) else "\n".join(request.prompt)
        extra = request.model_dump(exclude={"model", "prompt", "max_tokens", "temperature", "stream"})
        results = await self._route_text(prompt, request.max_tokens, request.temperature, extra)
        text = results[0] if results else ""
        return self._openai_completion_response(text, prompt)


    async def infer(self, request):
        """Programmatic inference entrypoint — shards batches by measured replica tokens/sec."""
        self._touch()
        if isinstance(request, dict):
            prompt = request.get("prompts") or request.get("prompt")
            kwargs = {k: v for k, v in request.items() if k not in ("prompt", "prompts")}
        else:
            prompt, kwargs = str(request), {}

        prompts = prompt if isinstance(prompt, list) else [prompt]
        return await self._dispatch_prompts(prompts, self._sampling_params(extra=kwargs))
