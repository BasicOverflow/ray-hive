"""
Model router — engine-queue load balancing with OpenAI-compatible HTTP ingress.

Exposes /v1/models, /v1/chat/completions, /v1/completions, /v1/embeddings.
Programmatic inference goes through infer() → replica handle.
"""
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from ray import serve
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from ray_hive.core.ray_utils.media import (
    audio_from_b64,
    audio_from_url,
    pil_from_url,
    video_frames_from_url,
)
from ray_hive.core.openai_protocol import (
    ToolCallStreamFilter,
    clamp_max_tokens,
    finish_reason_for,
    hf_tool_calls,
    model_card,
    normalize_role,
    normalize_tools,
    parse_tool_calls,
    template_apply_errors,
    template_message_encodings,
)
from ray_hive.core.think_split import ThinkStreamFilter, sanitize_stop, strip_think
from ray_hive.errors import MediaError, UnsupportedModeError, http_status_for


def _ensure_llama_flash_attention_compat() -> None:
    """DeepSeek-OCR* remote code still imports symbols removed in transformers 5.x."""
    try:
        import transformers.models.llama.modeling_llama as llama_mod
    except Exception:
        return
    if not hasattr(llama_mod, "LlamaFlashAttention2"):
        base = getattr(llama_mod, "LlamaAttention", None)
        if base is not None:
            llama_mod.LlamaFlashAttention2 = base
    try:
        import transformers.utils.import_utils as iu

        if not hasattr(iu, "is_torch_fx_available"):
            iu.is_torch_fx_available = lambda: False  # type: ignore[attr-defined]
    except Exception:
        pass


def _load_hf_tokenizer(model_name: str):
    """Load tokenizer; avoid DeepSeek remote modeling that breaks on transformers 5.x."""
    _ensure_llama_flash_attention_compat()
    # Prefer no remote code for DeepSeek — modeling_*.py is TF4-era and fails on TF5.
    prefer_remote = "deepseek" not in model_name.lower()
    order = (True, False) if prefer_remote else (False, True)
    last_err = None
    for remote in order:
        try:
            return AutoTokenizer.from_pretrained(model_name, trust_remote_code=remote)
        except Exception as e:
            last_err = e
            print(f"[ray-hive] AutoTokenizer(trust_remote_code={remote}) failed: {e}")
    raise last_err


def _default_deepseek_ocr_chat_template() -> str:
    """Plain DeepSeek-OCR prompt: <image>\\n{text} (no HF chat template on hub)."""
    return (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{% if message['content'] is string %}{{ message['content'] }}"
        "{% else %}"
        "{% for content in message['content'] %}"
        "{% if content['type'] == 'image' or content['type'] == 'image_url' %}<image>\n"
        "{% elif content['type'] == 'text' %}{{ content['text'] }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"
    )


def _ensure_chat_template(tokenizer, model_name: str, chat_template: str | None) -> None:
    if chat_template:
        tokenizer.chat_template = chat_template
        return
    if getattr(tokenizer, "chat_template", None):
        return
    name = model_name.lower()
    if "deepseek" in name and "ocr" in name:
        tokenizer.chat_template = _default_deepseek_ocr_chat_template()


def _image_placeholder_for_model(model_name: str) -> str:
    """Vision pad token(s) when flattening list content for brittle chat templates."""
    name = (model_name or "").lower()
    if "dots" in name and "ocr" in name:
        return "<|img|><|imgpad|><|endofimg|>"
    if "deepseek" in name and "ocr" in name:
        return "<image>\n"
    return "<image>\n"


def _flatten_mm_message_content(
    messages: list[dict],
    *,
    image_placeholder: str,
) -> list[dict]:
    """Collapse OpenAI-style multimodal list content to plain strings.

    Some OCR chat templates (notably dots.ocr) do ``'<|role|>' + m.content`` on
    system/assistant turns and break when any turn still has list content, or
    when processor preprocessing leaves list-shaped fields. Flattening keeps
    image pads so vLLM can still bind ``multi_modal_data``.
    """
    out: list[dict] = []
    for msg in messages:
        row = dict(msg)
        content = row.get("content")
        if isinstance(content, list):
            bits: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if (
                    ptype in ("image", "image_url")
                    or "image" in part
                    or "image_url" in part
                ):
                    bits.append(image_placeholder)
                elif ptype == "text" or "text" in part:
                    bits.append(str(part.get("text") or ""))
            row["content"] = "".join(bits)
        out.append(row)
    return out


def _load_mm_config(model_name: str):
    """Config for modality detection without executing broken remote modeling modules."""
    _ensure_llama_flash_attention_compat()
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"[ray-hive] AutoConfig(trust_remote_code=True) failed: {e}")
        from types import SimpleNamespace

        from huggingface_hub import hf_hub_download

        raw = json.loads(Path(hf_hub_download(model_name, "config.json")).read_text(encoding="utf-8"))

        def _ns(obj):
            if isinstance(obj, dict):
                return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
            return obj

        return _ns(raw)
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
_TEMPLATE_REQUEST_KEYS = (
    "chat_template_kwargs",
    "enable_thinking",
    "reasoning_effort",
    "preserve_thinking",
)

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
    """OpenAI chat content part — image URL or data URL."""
    type: Literal["image_url"]
    image_url: dict


class VideoUrlContentPart(BaseModel):
    """OpenAI chat content part — video URL or data URL."""
    type: Literal["video_url", "video"]
    video_url: dict | None = None
    video: dict | None = None


class AudioContentPart(BaseModel):
    """OpenAI chat content part — audio URL or input_audio."""
    type: Literal["audio_url", "input_audio"]
    audio_url: dict | None = None
    input_audio: dict | None = None


class FileContentPart(BaseModel):
    """OpenAI chat content part — file reference."""
    type: Literal["file"]
    file: dict


ContentPart = Union[
    TextContentPart,
    ImageUrlContentPart,
    VideoUrlContentPart,
    AudioContentPart,
    FileContentPart,
]


class ChatFunctionCall(BaseModel):
    """OpenAI tool function payload."""
    model_config = ConfigDict(extra="allow")
    name: str
    arguments: Union[str, dict] = "{}"


class ChatToolCall(BaseModel):
    """OpenAI assistant tool_calls item."""
    model_config = ConfigDict(extra="allow")
    id: str | None = None
    type: str = "function"
    function: ChatFunctionCall


class ChatMessage(BaseModel):
    """OpenAI chat message — content may be null when tool_calls are set."""
    model_config = ConfigDict(extra="allow")
    role: str
    content: Union[str, list[ContentPart], None] = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ChatToolCall] | None = None


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


class EmbeddingRequest(BaseModel):
    """OpenAI /v1/embeddings request (plus optional multimodal messages)."""
    model_config = ConfigDict(extra="allow")
    model: str
    input: Union[str, list[str], None] = None
    messages: list[ChatMessage] | None = None
    encoding_format: str | None = None


def _http_error(exc):
    raise HTTPException(status_code=http_status_for(exc), detail=str(exc)) from exc


def _audio_from_part(part: AudioContentPart):
    if part.type == "audio_url" and part.audio_url:
        return audio_from_url(part.audio_url.get("url") or "")
    if part.type == "input_audio" and part.input_audio:
        return audio_from_b64(part.input_audio.get("data") or "")
    _http_error(MediaError("invalid audio content part"))


def _video_url(part: VideoUrlContentPart) -> str:
    if part.video_url and part.video_url.get("url"):
        return part.video_url["url"]
    if part.video and part.video.get("url"):
        return part.video["url"]
    _http_error(MediaError("invalid video content part"))


@serve.deployment(
    ray_actor_options={"num_cpus": 0},
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
        pooling: bool = False,
        multimodal: bool = False,
        chat_template: str | None = None,
    ):
        """Wire replica handles, load cache, and tokenizer for token counting."""
        self.model_id = model_id
        self.model_name = model_name
        self.gpu_deployment_names = gpu_deployment_names
        self.replica_metadata = replica_metadata
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.idle_timeout = idle_timeout
        self.sleep_timeout = sleep_timeout
        self.pooling = pooling
        self.multimodal = multimodal
        self._handles = None
        self._loads = {name: {"waiting": 0, "running": 0} for name in gpu_deployment_names}
        self._eng_start = 0
        self._shutting_down = False
        self._sleeping = False
        self._sleep_lock = asyncio.Lock()
        self._template_style = None
        self.tokenizer = _load_hf_tokenizer(model_name)
        self.processor = None
        self._mm_warmup_modality = "image"
        if multimodal:
            from transformers import AutoProcessor

            cfg = _load_mm_config(model_name)
            if getattr(cfg, "audio_config", None) is not None and getattr(cfg, "vision_config", None) is None:
                self._mm_warmup_modality = "audio"
            elif getattr(cfg, "audio_config", None) is not None and getattr(cfg, "vision_config", None) is not None:
                self._mm_warmup_modality = "image"
            try:
                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            except Exception:
                try:
                    self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=False)
                except Exception:
                    self.processor = None
        _ensure_chat_template(self.tokenizer, model_name, chat_template)
        # DeepSeek-OCR* / dots.ocr: prefer tokenizer chat template. Processor
        # apply_chat_template has broken on list-shaped multimodal content.
        _mn = model_name.lower()
        if ("deepseek" in _mn and "ocr" in _mn) or ("dots" in _mn and "ocr" in _mn):
            self.processor = None
        if self.processor is not None and getattr(self.tokenizer, "chat_template", None):
            self.processor.chat_template = self.tokenizer.chat_template
        await self._warmup()
        self._last_activity = time.time()
        asyncio.create_task(self._refresh_loads())
        if self.idle_timeout > 0 or self.sleep_timeout > 0:
            asyncio.create_task(self._timeout_watch())


    def _apply_chat_template(
        self,
        hf_messages: list[dict],
        template_kwargs: dict | None = None,
        tools: list | None = None,
    ) -> str:
        """Apply this checkpoint's chat template; retry common tool encodings."""
        kwargs = dict(
            tokenize=False,
            add_generation_prompt=not self.pooling,
            **self.chat_template_kwargs,
        )
        if template_kwargs:
            kwargs.update(template_kwargs)
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            fn = self.processor.apply_chat_template
        else:
            fn = self.tokenizer.apply_chat_template

        encodings = template_message_encodings(hf_messages)
        # Retry with flattened MM content if list-shaped parts break Jinja.
        flat = _flatten_mm_message_content(
            hf_messages,
            image_placeholder=_image_placeholder_for_model(self.model_name),
        )
        if flat != hf_messages:
            encodings = list(encodings) + [("mm_flat_str", flat)]

        attempts: list[tuple[str, bool, list]] = []
        for name, msgs in encodings:
            if tools:
                attempts.append((name, True, msgs))
            attempts.append((name, False, msgs))

        cached = self._template_style
        if cached:
            preferred = [a for a in attempts if a[0] == cached[0] and a[1] == cached[1]]
            attempts = preferred + [a for a in attempts if a not in preferred]

        last = None
        errors = template_apply_errors()
        for name, use_tools, msgs in attempts:
            kw = dict(kwargs)
            if use_tools and tools:
                kw["tools"] = tools
            else:
                kw.pop("tools", None)
            try:
                prompt = fn(msgs, **kw)
            except errors as e:
                last = e
                continue
            self._template_style = (name, use_tools)
            return prompt
        if last is not None:
            raise last
        return fn(hf_messages, **kwargs)


    def _request_template_kwargs(self, extra: dict | None) -> dict:
        """Merge OpenWebUI chat_template_kwargs / enable_thinking into the template."""
        extra = extra or {}
        merged: dict = {}
        req = extra.get("chat_template_kwargs")
        if isinstance(req, dict):
            merged.update(req)
        for key in ("enable_thinking", "reasoning_effort", "preserve_thinking"):
            if extra.get(key) is not None:
                merged[key] = extra[key]
        return merged


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
                    await self._hold_vram_for_sleep()
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
            await self._release_vram_after_wake()
            self._sleeping = False


    async def _hold_vram_for_sleep(self):
        """Pending-hold planned VRAM before engine sleep so allocators cannot steal it."""
        import ray
        from ray_hive.core.gpu_registry import get_gpu_registry

        registry = get_gpu_registry()
        await asyncio.to_thread(
            ray.get, registry.mark_sleeping.remote(self.gpu_deployment_names)
        )


    async def _release_vram_after_wake(self):
        """Clear sleep pending hold after engines have reclaimed VRAM."""
        import ray
        from ray_hive.core.gpu_registry import get_gpu_registry

        registry = get_gpu_registry()
        await asyncio.to_thread(
            ray.get, registry.mark_awake.remote(self.gpu_deployment_names)
        )


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


    def _mm_warmup_prompt(self, text: str = "warmup") -> dict:
        """
        Build a PromptType dict with chat-template placeholders + dummy media.

        Bare text + multi_modal_data fails (no modality pads). Vision needs
        ≥~28×28 images; audio needs (array, sample_rate).
        """
        import numpy as np

        if self._mm_warmup_modality == "audio":
            audio = (np.zeros(16000, dtype=np.float32), 16000)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "audio"},
                    {"type": "text", "text": text},
                ],
            }]
            prompt = self._apply_chat_template(messages)
            return {"prompt": prompt, "multi_modal_data": {"audio": audio}}

        from PIL import Image

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }]
        prompt = self._apply_chat_template(messages)
        return {"prompt": prompt, "multi_modal_data": {"image": img}}


    async def _warmup(self):
        """Heat each replica engine with a fixed batch (no tok/s retained)."""
        handles = self._get_handles()
        if self.pooling:
            if self.multimodal:
                prompts = [self._mm_warmup_prompt()]
            else:
                prompts = [f"warmup {i}" for i in range(min(8, _WARMUP_PROMPTS))]
            for name in self.gpu_deployment_names:
                await handles[name].embed.remote(prompts)
            return

        if self.multimodal:
            prompts: list[Any] = [
                self._mm_warmup_prompt(f"warmup {i}")
                for i in range(min(4, _WARMUP_PROMPTS))
            ]
        else:
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


    def _parse_messages(self, messages: list[ChatMessage]) -> tuple[list[dict], dict]:
        """
        Convert OpenAI chat messages to HF-style messages + multi_modal_data.

        Returns (hf_messages, multi_modal_data).
        """
        hf_messages = []
        images = []
        videos = []
        audios = []

        for msg in messages:
            row: dict[str, Any] = {"role": normalize_role(msg.role)}
            if msg.name:
                row["name"] = msg.name
            if msg.tool_call_id:
                row["tool_call_id"] = msg.tool_call_id
            tc = hf_tool_calls(
                [c.model_dump() for c in msg.tool_calls] if msg.tool_calls else None
            )
            if tc:
                row["tool_calls"] = tc

            if msg.content is None:
                row["content"] = None if tc else ""
                hf_messages.append(row)
                continue
            if isinstance(msg.content, str):
                row["content"] = msg.content
                hf_messages.append(row)
                continue

            parts_out = []
            for part in msg.content:
                if part.type == "text":
                    parts_out.append({"type": "text", "text": part.text})
                elif part.type == "image_url":
                    url = (part.image_url or {}).get("url") or ""
                    images.append(pil_from_url(url))
                    parts_out.append({"type": "image"})
                elif part.type in ("video_url", "video"):
                    try:
                        videos.append(video_frames_from_url(_video_url(part)))
                    except MediaError as e:
                        _http_error(e)
                    parts_out.append({"type": "video"})
                elif part.type in ("audio_url", "input_audio"):
                    audios.append(_audio_from_part(part))
                    parts_out.append({"type": "audio"})
                elif part.type == "file":
                    _http_error(MediaError("file content parts not supported"))
                else:
                    _http_error(MediaError(f"unsupported content type: {part.type}"))

            if len(parts_out) == 1 and parts_out[0].get("type") == "text":
                row["content"] = parts_out[0]["text"]
            else:
                row["content"] = parts_out
            hf_messages.append(row)

        mm: dict[str, Any] = {}
        if images:
            mm["image"] = images[0] if len(images) == 1 else images
        if videos:
            mm["video"] = videos[0] if len(videos) == 1 else videos
        if audios:
            mm["audio"] = audios[0] if len(audios) == 1 else audios
        return hf_messages, mm


    def _chat_prompt_and_mm(
        self,
        messages: list[ChatMessage],
        template_kwargs: dict | None = None,
        tools: list | None = None,
    ) -> tuple[str, dict]:
        """Apply chat template; return (prompt_str, multi_modal_data)."""
        hf_messages, mm = self._parse_messages(messages)
        prompt = self._apply_chat_template(hf_messages, template_kwargs, tools=tools)
        return prompt, mm


    def _to_engine_prompt(self, prompt: str, mm: dict | None):
        """Build str or PromptType dict for the engine."""
        if mm:
            return {"prompt": prompt, "multi_modal_data": mm}
        return prompt


    def _is_openai_messages(self, obj: Any) -> bool:
        """True for a chat conversation: list of {role, content} dicts/models."""
        if not isinstance(obj, list) or not obj:
            return False
        first = obj[0]
        if isinstance(first, ChatMessage):
            return True
        return isinstance(first, dict) and "role" in first


    def _coerce_chat_messages(self, messages: list) -> list[ChatMessage]:
        return [
            m if isinstance(m, ChatMessage) else ChatMessage.model_validate(m)
            for m in messages
        ]


    def _normalize_prompt(self, item: Any) -> Any:
        """str | engine PromptType | OpenAI messages → engine prompt."""
        if self._is_openai_messages(item):
            prompt, mm = self._chat_prompt_and_mm(self._coerce_chat_messages(item))
            return self._to_engine_prompt(prompt, mm)
        if isinstance(item, dict) and "role" in item and "content" in item and "prompt" not in item:
            prompt, mm = self._chat_prompt_and_mm(self._coerce_chat_messages([item]))
            return self._to_engine_prompt(prompt, mm)
        return item


    def _normalize_infer_prompts(self, prompt: Any) -> list:
        """
        Expand infer() prompt/prompts into engine prompts.

        A single OpenAI conversation is a list of message dicts — that must not
        be treated as a batch of independent prompts.
        """
        if self._is_openai_messages(prompt):
            return [self._normalize_prompt(prompt)]
        if isinstance(prompt, list):
            return [self._normalize_prompt(p) for p in prompt]
        return [self._normalize_prompt(prompt)]


    async def _route_chat(
        self,
        messages,
        max_tokens=None,
        temperature=None,
        extra=None,
        template_kwargs=None,
        tools=None,
    ):
        """Route a chat conversation to the least-loaded replica."""
        prompt, mm = self._chat_prompt_and_mm(messages, template_kwargs, tools=tools)
        engine_prompt = self._to_engine_prompt(prompt, mm)
        replica_name = self._select_replica()
        handle = self._get_handles()[replica_name]
        return await handle.generate.remote(
            [engine_prompt],
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
        kwargs = {k: v for k, v in dict(extra or {}).items() if v is not None}
        # OpenAI request-only fields — not SamplingParams (OpenWebUI sends these).
        for k in (
            "stream_options", "user", "tools", "tool_choice", "functions", "function_call",
            "metadata", "modalities", "audio", "service_tier", "store", "parallel_tool_calls",
            "prediction", "web_search_options", "logit_bias", "logprobs", "top_logprobs",
            *_TEMPLATE_REQUEST_KEYS,
        ):
            kwargs.pop(k, None)
        if "stop" in kwargs:
            kwargs["stop"] = sanitize_stop(kwargs["stop"])
            if kwargs["stop"] is None:
                kwargs.pop("stop")
        mct = kwargs.pop("max_completion_tokens", None)
        requested = max_tokens if max_tokens is not None else mct
        clamped = clamp_max_tokens(requested, self.replica_metadata)
        if clamped is not None:
            kwargs["max_tokens"] = clamped
        if temperature is not None:
            kwargs["temperature"] = temperature
        guided_json = kwargs.pop("guided_json", None)
        response_format = kwargs.pop("response_format", None)
        vllm_xargs = kwargs.pop("vllm_xargs", None)
        if vllm_xargs:
            merged = dict(kwargs.get("extra_args") or {})
            merged.update(vllm_xargs)
            kwargs["extra_args"] = merged
        if guided_json is None and isinstance(response_format, dict):
            if response_format.get("type") == "json_schema":
                guided_json = (response_format.get("json_schema") or {}).get("schema")
        if guided_json is not None:
            kwargs["structured_outputs"] = StructuredOutputsParams(json=guided_json)
        # Keep only fields SamplingParams accepts (msgspec Struct).
        valid = set(getattr(SamplingParams, "__struct_fields__", ()))
        if valid:
            kwargs = {k: v for k, v in kwargs.items() if k in valid}
        return SamplingParams(**kwargs)


    def _extract_texts(self, outputs) -> list[str]:
        """Extract generated text strings from vLLM RequestOutput list."""
        return [o.outputs[0].text for o in outputs]


    def _shard_prompts(self, prompts: list) -> list[tuple[str, list, list[int]]]:
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


    async def _dispatch_prompts(self, prompts: list, sampling_params: SamplingParams) -> list[str]:
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


    async def _dispatch_embeds(self, prompts: list) -> list[list[float]]:
        """Shard embed prompts across replicas."""
        handles = self._get_handles()
        if len(prompts) == 1:
            name = self._select_replica()
            return await handles[name].embed.remote(prompts)

        shards = self._shard_prompts(prompts)
        for name, chunk, _ in shards:
            self._loads[name]["waiting"] += len(chunk)
        outputs = await asyncio.gather(*[
            handles[name].embed.remote(chunk)
            for name, chunk, _ in shards
        ])
        vectors = [None] * len(prompts)
        for (_, _, idxs), outs in zip(shards, outputs):
            for local_i, global_i in enumerate(idxs):
                vectors[global_i] = outs[local_i]
        return vectors


    async def _route_text(self, prompt, max_tokens=None, temperature=None, extra=None):
        """Route text prompts across replicas by capacity."""
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        return await self._dispatch_prompts(
            prompts,
            self._sampling_params(max_tokens, temperature, extra),
        )


    async def _route_stream(self, prompt, sampling_params: SamplingParams) -> AsyncIterator[str]:
        """Stream text deltas from the least-loaded replica."""
        replica_name = self._select_replica()
        handle = self._get_handles()[replica_name].options(stream=True)
        async for delta in handle.generate_stream.remote(prompt, sampling_params):
            yield delta


    def _sse(self, payload: dict | str) -> str:
        """Format one SSE data line (dict → JSON, or raw string like [DONE])."""
        if isinstance(payload, str):
            return f"data: {payload}\n\n"
        return f"data: {json.dumps(payload)}\n\n"


    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))


    def _usage(self, prompt: str, completion: str) -> dict:
        prompt_tokens = self._count_tokens(prompt)
        completion_tokens = self._count_tokens(completion)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }


    def _model_card(self) -> dict:
        return model_card(self.model_id, self.replica_metadata, self.chat_template_kwargs)


    def _openai_chat_response(
        self,
        text: str | None,
        prompt: str,
        tool_calls: list | None = None,
        finish_reason: str = "stop",
        raw_text: str | None = None,
    ) -> dict:
        """Build OpenAI chat completion response dict."""
        message = {"role": "assistant", "content": text}
        if tool_calls:
            message["tool_calls"] = tool_calls
            if text is None:
                message["content"] = None
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "model": self.model_id,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": self._usage(prompt, raw_text if raw_text is not None else (text or "")),
        }


    def _openai_completion_response(self, text: str, prompt: str, finish_reason: str = "stop") -> dict:
        """Build OpenAI text completion response dict."""
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "model": self.model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
            "usage": self._usage(prompt, text),
        }


    def _openai_embedding_response(self, vectors: list[list[float]]) -> dict:
        """Build OpenAI embeddings response dict."""
        data = [
            {"object": "embedding", "embedding": vec, "index": i}
            for i, vec in enumerate(vectors)
        ]
        return {
            "object": "list",
            "data": data,
            "model": self.model_id,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }


    @app.get("/v1/models")
    async def list_models(self):
        """OpenAI-compatible model list endpoint."""
        return {"object": "list", "data": [self._model_card()]}


    @app.get("/v1/models/{model_id}")
    async def get_model(self, model_id: str):
        """OpenAI-compatible single model endpoint."""
        return self._model_card()


    def _chat_chunk(self, chunk_id: str, delta: dict, finish_reason=None, usage=None) -> str:
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "model": self.model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            payload["usage"] = usage
        return self._sse(payload)


    async def _openai_chat_stream(
        self,
        prompt,
        sampling_params: SamplingParams,
        prompt_text: str = "",
        include_usage: bool = False,
    ):
        """Yield OpenAI chat.completion.chunk SSE frames then [DONE]."""
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        yield self._chat_chunk(chunk_id, {"role": "assistant", "content": ""})
        think = ThinkStreamFilter()
        tools = ToolCallStreamFilter()
        visible = []
        raw_bits = []
        async for delta in self._route_stream(prompt, sampling_params):
            raw_bits.append(delta)
            piece = think.feed(delta)
            if not piece:
                continue
            emit = tools.feed(piece)
            if emit:
                visible.append(emit)
                yield self._chat_chunk(chunk_id, {"content": emit})
        tail_think = think.flush()
        if tail_think:
            emit = tools.feed(tail_think)
            if emit:
                visible.append(emit)
                yield self._chat_chunk(chunk_id, {"content": emit})
        held, tool_calls = tools.flush()
        if held and not tool_calls:
            visible.append(held)
            yield self._chat_chunk(chunk_id, {"content": held})
        elif held and tool_calls:
            visible.append(held)
            yield self._chat_chunk(chunk_id, {"content": held})
        if tool_calls:
            yield self._chat_chunk(chunk_id, {"tool_calls": tool_calls})
            finish = "tool_calls"
        else:
            finish = "length" if (
                sampling_params.max_tokens
                and self._count_tokens("".join(raw_bits)) >= sampling_params.max_tokens
            ) else "stop"
        usage = self._usage(prompt_text, "".join(raw_bits))
        yield self._chat_chunk(chunk_id, {}, finish_reason=finish, usage=usage)
        if include_usage:
            yield self._sse({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "model": self.model_id,
                "choices": [],
                "usage": usage,
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
        """OpenAI-compatible chat completions (text + multimodal + tools)."""
        if self.pooling:
            _http_error(UnsupportedModeError("this deployment is pooling/embed; use /v1/embeddings"))
        self._touch()
        await self._ensure_awake()
        extra = request.model_dump(exclude={"model", "messages", "max_tokens", "temperature", "stream"})
        tmpl = self._request_template_kwargs(extra)
        tools = normalize_tools(extra.get("tools"), extra.get("functions"))
        params = self._sampling_params(request.max_tokens, request.temperature, extra)
        prompt, mm = self._chat_prompt_and_mm(request.messages, tmpl, tools=tools)
        engine_prompt = self._to_engine_prompt(prompt, mm)
        include_usage = bool((extra.get("stream_options") or {}).get("include_usage"))
        if request.stream:
            return StreamingResponse(
                self._openai_chat_stream(
                    engine_prompt, params, prompt_text=prompt, include_usage=include_usage,
                ),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        outputs = await self._route_chat(
            request.messages, request.max_tokens, request.temperature, extra, tmpl, tools,
        )
        raw = self._extract_texts(outputs)[0] if outputs else ""
        text = strip_think(raw)
        content, tool_calls = parse_tool_calls(text)
        finish = finish_reason_for(outputs[0] if outputs else None, tool_calls, params.max_tokens)
        return self._openai_chat_response(
            content, prompt, tool_calls=tool_calls or None, finish_reason=finish, raw_text=raw,
        )


    @app.post("/v1/completions")
    async def completions(self, request: CompletionRequest):
        """OpenAI-compatible text completions endpoint."""
        if self.pooling:
            _http_error(UnsupportedModeError("this deployment is pooling/embed; use /v1/embeddings"))
        self._touch()
        await self._ensure_awake()
        prompt = request.prompt if isinstance(request.prompt, str) else "\n".join(request.prompt)
        extra = request.model_dump(exclude={"model", "prompt", "max_tokens", "temperature", "stream"})
        params = self._sampling_params(request.max_tokens, request.temperature, extra)
        if request.stream:
            return StreamingResponse(
                self._openai_completion_stream(prompt, params),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        results = await self._route_text(prompt, request.max_tokens, request.temperature, extra)
        text = results[0] if results else ""
        return self._openai_completion_response(text, prompt)


    @app.post("/v1/embeddings")
    async def embeddings(self, request: EmbeddingRequest):
        """OpenAI-compatible embeddings endpoint (text and multimodal messages)."""
        if not self.pooling:
            _http_error(UnsupportedModeError("this deployment is generate-only; set runner=pooling"))
        self._touch()
        await self._ensure_awake()

        prompts: list[Any] = []
        if request.messages is not None:
            prompt, mm = self._chat_prompt_and_mm(request.messages)
            prompts.append(self._to_engine_prompt(prompt, mm))
        elif request.input is None:
            raise HTTPException(status_code=400, detail="input or messages required")
        elif isinstance(request.input, str):
            prompts.append(request.input)
        else:
            prompts.extend(request.input)

        vectors = await self._dispatch_embeds(prompts)
        return self._openai_embedding_response(vectors)


    async def infer(self, request):
        """Programmatic inference — generate text or embeddings depending on deploy mode."""
        self._touch()
        await self._ensure_awake()
        if isinstance(request, dict):
            prompt = request.get("prompts") or request.get("prompt")
            kwargs = {k: v for k, v in request.items() if k not in ("prompt", "prompts")}
        else:
            prompt, kwargs = request, {}

        prompts = self._normalize_infer_prompts(prompt)

        if self.pooling:
            if kwargs.get("guided_json") is not None or kwargs.get("structured_output") is not None:
                _http_error(UnsupportedModeError("structured_output not supported for embeddings"))
            return await self._dispatch_embeds(prompts)

        return await self._dispatch_prompts(prompts, self._sampling_params(extra=kwargs))
