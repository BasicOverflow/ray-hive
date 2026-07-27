"""
Model router — engine-queue load balancing with OpenAI-compatible HTTP ingress.

Exposes /v1/models, /v1/chat/completions, /v1/completions on the router
deployment. Programmatic inference goes through infer() → replica handle.
"""
import asyncio
import json
import time
import uuid
from typing import AsyncIterator, Literal, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from ray import serve
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

app = FastAPI()

_WARMUP_PROMPTS = 32
_WARMUP_MAX_TOKENS = 32
_LOAD_REFRESH_S = 0.1
_WAITING_WEIGHT = 4


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
    """Router with engine-queue balancing and OpenAI-compatible HTTP ingress."""

    async def __init__(
        self,
        model_id: str,
        model_name: str,
        gpu_deployment_names: list[str],
        replica_metadata: dict,
        chat_template_kwargs: dict | None = None,
        idle_timeout: int = -1,
        sleep_timeout: int = -1,
    ):
        """Wire replica handles, load cache, and tokenizer for token counting."""
        self.model_id = model_id
        self.model_name = model_name
        self.gpu_deployment_names = gpu_deployment_names
        self.replica_metadata = replica_metadata
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.idle_timeout = idle_timeout
        self.sleep_timeout = sleep_timeout
        self._handles = None
        self._loads = {name: {"waiting": 0, "running": 0} for name in gpu_deployment_names}
        self._eng_start = 0
        self._shutting_down = False
        self._sleeping = False
        self._sleep_lock = asyncio.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        await self._warmup()
        self._last_activity = time.time()
        asyncio.create_task(self._refresh_loads())
        if self.idle_timeout > 0 or self.sleep_timeout > 0:
            asyncio.create_task(self._timeout_watch())


    def _touch(self):
        """Record inference activity for sleep/idle timeouts."""
        self._last_activity = time.time()


    async def _timeout_watch(self):
        """Sleep replicas after sleep_timeout; destroy model after idle_timeout."""
        from ray_hive.core.ray_utils.lifecycle import shutdown_model

        active = [t for t in (self.sleep_timeout, self.idle_timeout) if t > 0]
        interval = min(5, *active)
        while not self._shutting_down:
            await asyncio.sleep(interval)
            quiet = time.time() - self._last_activity
            if self.idle_timeout > 0 and quiet >= self.idle_timeout:
                self._shutting_down = True
                await asyncio.to_thread(shutdown_model, self.model_id)
                return
            if (
                self.sleep_timeout > 0
                and not self._sleeping
                and quiet >= self.sleep_timeout
            ):
                async with self._sleep_lock:
                    if self._sleeping or self._shutting_down:
                        continue
                    quiet = time.time() - self._last_activity
                    if quiet < self.sleep_timeout:
                        continue
                    self._sleeping = True
                    handles = self._get_handles()
                    await asyncio.gather(*[
                        handles[name].sleep.remote(1)
                        for name in self.gpu_deployment_names
                    ])


    async def _ensure_awake(self):
        """Wake all replicas if sleeping (call after _touch on real inference)."""
        if self.sleep_timeout <= 0:
            return
        async with self._sleep_lock:
            if not self._sleeping:
                return
            handles = self._get_handles()
            await asyncio.gather(*[
                handles[name].wake_up.remote()
                for name in self.gpu_deployment_names
            ])
            self._sleeping = False


    async def _refresh_loads(self):
        """Poll replica engine queue depths into the local load cache."""
        handles = self._get_handles()
        while not self._shutting_down:
            results = await asyncio.gather(*[
                handles[name].get_load.remote()
                for name in self.gpu_deployment_names
            ])
            for name, load in zip(self.gpu_deployment_names, results):
                self._loads[name] = load
            await asyncio.sleep(_LOAD_REFRESH_S)


    async def _warmup(self):
        """Heat each replica engine with a fixed batch (no tok/s retained)."""
        handles = self._get_handles()
        prompts = [f"warmup {i}" for i in range(_WARMUP_PROMPTS)]
        params = SamplingParams(max_tokens=_WARMUP_MAX_TOKENS, temperature=0.0)
        for name in self.gpu_deployment_names:
            await handles[name].generate.remote(prompts, params)


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
        handle = self._get_handles()[replica_name]
        return await handle.generate.remote(
            [prompt],
            self._sampling_params(max_tokens, temperature, extra),
        )


    def _select_replica(self) -> str:
        """Pick replica with lowest (waiting*4+running) / max_num_seqs."""
        names = self.gpu_deployment_names
        n = len(names)
        best_name = None
        best_score = float("inf")
        for i in range(n):
            name = names[(self._eng_start + i) % n]
            load = self._loads[name]
            cap = max(self.replica_metadata[name]["max_num_seqs"], 1)
            score = (load["waiting"] * _WAITING_WEIGHT + load["running"]) / cap
            if score < best_score:
                best_score = score
                best_name = name
        self._loads[best_name]["waiting"] += 1
        self._eng_start = (self._eng_start + 1) % n
        return best_name


    def _sampling_params(self, max_tokens=None, temperature=None, extra=None) -> SamplingParams:
        """Build vLLM SamplingParams from request fields."""
        kwargs = dict(extra or {})
        # OpenAI/LangChain send max_completion_tokens; SamplingParams wants max_tokens
        mct = kwargs.pop("max_completion_tokens", None)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        elif mct is not None:
            kwargs["max_tokens"] = mct
        if temperature is not None:
            kwargs["temperature"] = temperature
        guided_json = kwargs.pop("guided_json", None)
        response_format = kwargs.pop("response_format", None)
        if guided_json is None and isinstance(response_format, dict):
            if response_format.get("type") == "json_schema":
                guided_json = (response_format.get("json_schema") or {}).get("schema")
        if guided_json is not None:
            kwargs["structured_outputs"] = StructuredOutputsParams(json=guided_json)
        return SamplingParams(**kwargs)


    def _extract_texts(self, outputs) -> list[str]:
        """Extract generated text strings from vLLM RequestOutput list."""
        return [o.outputs[0].text for o in outputs]


    def _shard_prompts(self, prompts: list[str]) -> list[tuple[str, list[str], list[int]]]:
        """Split prompts by max_num_seqs; largest-remainder for leftovers."""
        names = self.gpu_deployment_names
        weights = [max(self.replica_metadata[n]["max_num_seqs"], 1) for n in names]
        total_w = sum(weights)
        n = len(prompts)

        quotas = [n * w / total_w for w in weights]
        counts = [int(q) for q in quotas]
        leftover = n - sum(counts)
        order = sorted(
            range(len(names)),
            key=lambda i: (quotas[i] - counts[i], weights[i]),
            reverse=True,
        )
        for i in order[:leftover]:
            counts[i] += 1

        shards = []
        start = 0
        for name, count in zip(names, counts):
            end = start + count
            if count > 0:
                shards.append((name, prompts[start:end], list(range(start, end))))
            start = end
        return shards


    async def _dispatch_prompts(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        """Shard a prompt batch by max_num_seqs and run replicas concurrently."""
        handles = self._get_handles()
        if len(prompts) == 1:
            name = self._select_replica()
            outputs = await handles[name].generate.remote(prompts, sampling_params)
            return self._extract_texts(outputs)

        shards = self._shard_prompts(prompts)
        for name, chunk, _ in shards:
            self._loads[name]["waiting"] += len(chunk)
        outputs = await asyncio.gather(*[
            handles[name].generate.remote(chunk, sampling_params)
            for name, chunk, _ in shards
        ])
        texts = [None] * len(prompts)
        for (_, _, idxs), outs in zip(shards, outputs):
            for local_i, global_i in enumerate(idxs):
                texts[global_i] = outs[local_i].outputs[0].text
        return texts


    async def _route_text(self, prompt, max_tokens=None, temperature=None, extra=None):
        """Route text prompts across replicas by capacity."""
        prompts = [prompt] if isinstance(prompt, str) else [str(p) for p in prompt]
        return await self._dispatch_prompts(
            prompts,
            self._sampling_params(max_tokens, temperature, extra),
        )


    async def _route_stream(self, prompt: str, sampling_params: SamplingParams) -> AsyncIterator[str]:
        """Stream text deltas from the least-loaded replica."""
        replica_name = self._select_replica()
        handle = self._get_handles()[replica_name].options(stream=True)
        async for delta in handle.generate_stream.remote(prompt, sampling_params):
            yield delta


    def _chat_prompt(self, messages) -> str:
        """Apply chat template to messages."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )


    def _sse(self, payload: dict | str) -> str:
        """Format one SSE data line (dict → JSON, or raw string like [DONE])."""
        if isinstance(payload, str):
            return f"data: {payload}\n\n"
        return f"data: {json.dumps(payload)}\n\n"


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


    async def _openai_chat_stream(self, prompt: str, sampling_params: SamplingParams):
        """Yield OpenAI chat.completion.chunk SSE frames then [DONE]."""
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        async for delta in self._route_stream(prompt, sampling_params):
            yield self._sse({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "model": self.model_id,
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
            })
        yield self._sse({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "model": self.model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })
        yield self._sse("[DONE]")


    async def _openai_completion_stream(self, prompt: str, sampling_params: SamplingParams):
        """Yield OpenAI text_completion SSE frames then [DONE]."""
        chunk_id = f"cmpl-{uuid.uuid4().hex}"
        async for delta in self._route_stream(prompt, sampling_params):
            yield self._sse({
                "id": chunk_id,
                "object": "text_completion",
                "model": self.model_id,
                "choices": [{"index": 0, "text": delta, "finish_reason": None}],
            })
        yield self._sse({
            "id": chunk_id,
            "object": "text_completion",
            "model": self.model_id,
            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
        })
        yield self._sse("[DONE]")


    @app.post("/v1/chat/completions")
    async def chat_completions(self, request: ChatCompletionRequest):
        """OpenAI-compatible chat completions endpoint (text-only)."""
        self._touch()
        await self._ensure_awake()
        messages = self._to_chat_messages(request.messages)
        extra = request.model_dump(exclude={"model", "messages", "max_tokens", "temperature", "stream"})
        params = self._sampling_params(request.max_tokens, request.temperature, extra)
        if request.stream:
            prompt = self._chat_prompt(messages)
            return StreamingResponse(
                self._openai_chat_stream(prompt, params),
                media_type="text/event-stream",
            )
        outputs = await self._route_chat(messages, request.max_tokens, request.temperature, extra)
        text = self._extract_texts(outputs)[0] if outputs else ""
        prompt = outputs[0].prompt if outputs else ""
        return self._openai_chat_response(text, prompt)


    @app.post("/v1/completions")
    async def completions(self, request: CompletionRequest):
        """OpenAI-compatible text completions endpoint."""
        self._touch()
        await self._ensure_awake()
        prompt = request.prompt if isinstance(request.prompt, str) else "\n".join(request.prompt)
        extra = request.model_dump(exclude={"model", "prompt", "max_tokens", "temperature", "stream"})
        params = self._sampling_params(request.max_tokens, request.temperature, extra)
        if request.stream:
            return StreamingResponse(
                self._openai_completion_stream(prompt, params),
                media_type="text/event-stream",
            )
        results = await self._route_text(prompt, request.max_tokens, request.temperature, extra)
        text = results[0] if results else ""
        return self._openai_completion_response(text, prompt)


    async def infer(self, request):
        """Programmatic inference entrypoint — shards batches by planned max_num_seqs."""
        self._touch()
        await self._ensure_awake()
        if isinstance(request, dict):
            prompt = request.get("prompts") or request.get("prompt")
            kwargs = {k: v for k, v in request.items() if k not in ("prompt", "prompts")}
        else:
            prompt, kwargs = str(request), {}

        prompts = prompt if isinstance(prompt, list) else [prompt]
        return await self._dispatch_prompts(prompts, self._sampling_params(extra=kwargs))
