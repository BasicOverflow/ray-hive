# Ray Hive

Ray Hive is a small control layer for running a heterogeneous Ray cluster on k3s and placing vLLM model replicas according to live GPU memory.

The cluster remains a normal Ray cluster: CPU workers can run general distributed tasks while GPU workers host model deployments through Ray Serve.

## Contents

- [Architecture](#architecture)
- [Capabilities](#capabilities)
- [Quick start](#quick-start)
- [Placement](#placement)
- [Multimodal and context](#multimodal-and-context)
- [Embeddings](#embeddings)
- [Lifecycle](#lifecycle)
- [OpenAI HTTP](#openai-http)
- [General Ray CPU Tasks](#general-ray-cpu-tasks)
- [Appendix](#appendix)

## Architecture

```text
Clients
  └─ Ray head
      ├─ CPU workers → general Ray tasks
      └─ GPU workers → vLLM replicas
           ├─ VRAM monitor DaemonSet
           ├─ shared GPU registry
           └─ Ray Serve model routers
```

KubeRay manages the head and worker pods. Ray handles task scheduling and Serve deployments; Ray Hive adds model planning, per-GPU placement, memory reservations, and routing.

## Capabilities

- **Throughput-first planning** — max practical `max_num_seqs` / batched tokens within a VRAM budget (`gpu_budget_frac` 0.90).
- **Heterogeneous replicas** — per-GPU plans so different cards each contribute what they can.
- **Least-loaded routing** — relative to each replica’s planned capacity.
- **Live VRAM scheduling** — registry + reservations; see `examples/4_test_allocation_policies.py`.
- **GPU sharing** — co-locate when needed; intentional share via same `gpu=` pin (`examples/6_shared_gpu.py`).
- **Flexible placement** — pin, N replicas, or `replicas=-1` (`examples/1_test_model_configs.py`).
- **Same-node TP** — auto escalate or pin a GPU list (`examples/5_tensor_parallel.py`).
- **Custom attention** — subclass for KV / MM token math (`examples/3_custom_attention.py`).
- **Multimodal generate** — image / video / audio (`examples/8`–`9`, `12`–`14`).
- **Embeddings** — `runner="pooling"` (`examples/10`–`11`).
- **Sleep / idle** — level-1 sleep then optional self-destroy (`examples/7_sleep_idle_timeout.py`).
- **OpenAI HTTP** — per-model routes + cluster `/v1` gateway (`examples/2_test_inference.py`).
- **General Ray tasks** — same cluster for CPU work.

**Out of scope:** precomputed multimodal embeds as *input* (`enable_mm_embeds`).

## Quick start

Dry-run packing against live GPUs (same plan deploy would use):

```python
from ray_hive import RayHive

hive = RayHive(address="ray://YOUR_RAY_HEAD_IP:10001")
plan = hive.estimate_vram(
    "Qwen/Qwen3-0.6B-FP8",
    max_input_prompt_length=1024,
    max_output_prompt_length=2048,
    replicas=1,
    vllm_kwargs={
        "trust_remote_code": True,
        "reasoning_parser": "qwen3",
        "default_chat_template_kwargs": {"enable_thinking": False},
    },
)
```

```text
Deployment Plan: Qwen/Qwen3-0.6B-FP8
  Replica                  replica-0
  GPU(s)                   host-a:gpu0
  tensor_parallel_size     1
  max_num_seqs             48
  max_num_batched_tokens   8192
  gpu_memory_utilization   0.850
  Weights                  0.60 GiB
  KV cache                 4.20 GiB
  Activations              0.40 GiB
  Overhead                 0.30 GiB
  Total (per GPU)          5.50 GiB
```

Deploy until ready (router up + warmed):

```python
status = hive.deploy_model(
    model_id="qwen",
    model_name="Qwen/Qwen3-0.6B-FP8",
    max_input_prompt_length=1024,
    max_output_prompt_length=2048,
    replicas=1,
    vllm_kwargs={
        "trust_remote_code": True,
        "reasoning_parser": "qwen3",
        "default_chat_template_kwargs": {"enable_thinking": False},
    },
)
```

```text
{"model_id": "qwen", "status": "ready", "route": "/qwen", "openai_v1": "/v1", "replicas": {...}}
```

Generate text (batch and stream helpers available):

```python
from ray_hive.inference import inference, inference_batch, inference_stream

answer = inference("Explain Ray in one sentence.", model_id="qwen")
answers = inference_batch(["Prompt one", "Prompt two"], model_id="qwen")
for delta in inference_stream("Count to three.", model_id="qwen"):
    print(delta, end="")
```

```text
Ray is a distributed computing framework for scaling Python workloads.
```

Structured output:

```python
from pydantic import BaseModel

class Answer(BaseModel):
    summary: str
    confidence: float

result = inference("Summarize Ray.", model_id="qwen", structured_output=Answer)
```

Inside an asyncio loop use `a_inference` / `a_inference_batch` instead of the sync helpers.

## Placement

- `gpu="host:gpu0"` — one replica on one GPU.
- `gpu=[a,b,...]` + `replicas=len(list)` — N single-GPU pins (TP=1 each).
- `gpu=[a,b,...]` + `replicas=1` — one same-node TP group.
- `gpu=None` — auto place (single GPU via `allocation_cls`, else same-node TP).
- `replicas=-1` — every eligible GPU / TP group.

Pin one GPU:

```python
hive.deploy_model(
    model_id="qwen-pin",
    model_name="Qwen/Qwen3-0.6B-FP8",
    max_input_prompt_length=512,
    max_output_prompt_length=512,
    replicas=1,
    gpu="ergos-06-nv:gpu0",
    vllm_kwargs={"trust_remote_code": True, "reasoning_parser": "qwen3",
                 "default_chat_template_kwargs": {"enable_thinking": False}},
)
```

All eligible GPUs:

```python
hive.deploy_model(
    model_id="qwen-all",
    model_name="Qwen/Qwen3-0.6B-FP8",
    max_input_prompt_length=512,
    max_output_prompt_length=512,
    replicas=-1,
    vllm_kwargs={"trust_remote_code": True, "reasoning_parser": "qwen3",
                 "default_chat_template_kwargs": {"enable_thinking": False}},
)
```

Same-node TP=2 pin:

```python
hive.deploy_model(
    model_id="qwen-tp",
    model_name="Qwen/Qwen3-8B-FP8",
    max_input_prompt_length=512,
    max_output_prompt_length=512,
    replicas=1,
    gpu=["ergos-02-nv:gpu1", "ergos-02-nv:gpu2"],
    vllm_kwargs={"trust_remote_code": True, "reasoning_parser": "qwen3",
                 "default_chat_template_kwargs": {"enable_thinking": False}},
)
```

Live registry snapshot: `hive.get_vram_state()`. Full TP / policy rules → [Appendix](#appendix).

## Multimodal and context

**Text-only models** — pass a string (or text chat messages). No `limit_mm_per_prompt`.

**MM models** — planner uses `limit_mm_per_prompt` in `vllm_kwargs` to size worst-case image / video / audio placeholders. If omitted on an MM HF config, defaults are derived from `vision_config` / `audio_config` (typically `image: 1` and/or `audio: 1`). Set counts explicitly to enable or disable modalities.

**Token budget** — `max_input_prompt_length` is the **text** side only. Effective input ≈ text + MM placeholders; `max_model_len ≈ effective_input + max_output_prompt_length` (use output `0` for pooling). If that cannot cover placeholders + output, planning raises `MmContextError` — raise `max_input_prompt_length` or lower `limit_mm_per_prompt`.

**Requests vs planning** — on an MM deploy you can still send text-only strings. That does **not** shrink the VRAM plan. For text-only *planning* on an MM checkpoint, zero unused modalities (`{"image": 0, "video": 0, "audio": 0}`).

Enable image chat:

```python
from ray_hive.core.ray_utils import file_to_data_url
from ray_hive.inference import inference

hive.deploy_model(
    model_id="vl",
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    max_input_prompt_length=2048,   # text tokens you expect
    max_output_prompt_length=256,
    replicas=1,
    vllm_kwargs={
        "trust_remote_code": True,
        "limit_mm_per_prompt": {"image": 1},
    },
)

messages = [{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": file_to_data_url("photo.png")}},
        {"type": "text", "text": "Describe this image."},
    ],
}]
answer = inference(messages, model_id="vl", max_tokens=64)
# text-only request on the same deploy still works:
inference("Say hello.", model_id="vl", max_tokens=32)
```

```text
# estimate_vram on MM models also shows:
  mm_tokens_per_prompt     1280
```

Content part types: `text`, `image_url`, `video_url` / `video`, `audio_url` / `input_audio`. Fixtures: `examples/media/`. Demos: `examples/8_multimodal_vision.py`, `9_multimodal_audio.py`, `12_multimodal_video.py`, `13_gemma4_multimodal.py`, `14_gemma4_stress.py`.

## Embeddings

Set `runner="pooling"` (legacy `task="embed"` still works). Use `max_output_prompt_length=0`. `inference` returns a vector (or list of vectors for batch).

```python
hive.deploy_model(
    model_id="embed",
    model_name="BAAI/bge-small-en-v1.5",
    max_input_prompt_length=512,
    max_output_prompt_length=0,
    replicas=1,
    vllm_kwargs={"runner": "pooling", "trust_remote_code": True},
)
vec = inference("hello", model_id="embed")
```

```text
[-0.012, 0.034, ...]   # length = model hidden size
```

```bash
curl http://YOUR_RAY_HEAD_IP:8000/embed/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"embed","input":["hello world"]}'
```

See `examples/10_text_embeddings.py` and `examples/11_multimodal_embeddings.py`.

## Lifecycle

Sleep after quiet, then optional full destroy (`idle_timeout` must be greater than `sleep_timeout` when both are set):

```python
hive.deploy_model(
    model_id="qwen",
    model_name="Qwen/Qwen3-0.6B-FP8",
    max_input_prompt_length=512,
    max_output_prompt_length=256,
    replicas=1,
    sleep_timeout=60,
    idle_timeout=300,
    vllm_kwargs={"trust_remote_code": True, "reasoning_parser": "qwen3",
                 "default_chat_template_kwargs": {"enable_thinking": False}},
)
```

```python
hive.shutdown("qwen")   # one model
hive.shutdown()         # all models
```

```python
from ray_hive import kill_gpu_registry
kill_gpu_registry()     # force registry rebuild (DaemonSet re-registers)
```

See `examples/7_sleep_idle_timeout.py` and `examples/0_shutdown_models.py`.

## OpenAI HTTP

Per-model route and cluster-wide `/v1` gateway (port `8000`). Gateway starts on `deploy_model`; for an already-running cluster call `hive.ensure_openai_api()`.

```bash
curl http://YOUR_RAY_HEAD_IP:8000/qwen/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Explain Ray briefly."}]}'

# Open WebUI: API URL = http://YOUR_RAY_HEAD_IP:8000/v1
curl http://YOUR_RAY_HEAD_IP:8000/v1/models
curl http://YOUR_RAY_HEAD_IP:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Explain Ray briefly."}]}'
```

Streaming, structured JSON, and LangChain: `examples/2_test_inference.py`.

## General Ray CPU Tasks

```python
import ray

ray.init(address="ray://YOUR_RAY_HEAD_IP:10001")

@ray.remote
def square(value):
    return value * value

results = ray.get([square.remote(value) for value in range(10)])
```

### Allocation policies

When `gpu=` is unset, `allocation_cls` picks GPUs (`ray_hive.core.gpu_alloc` / `ray_gpu_alloc`). Pins skip policy ranking but still check VRAM fit and arch taints.

- `RayPerformanceAllocator` — rank by compute proxy; top-N.
- `RayConserveTdpAllocator` — prefer lower approx TDP; SM count tie-break.
- `RayTensorParallelAllocator` — same-node packs; auto when single-GPU fails.

**Arch taint:** native FP8 (HF / vLLM dtype / kv / quantization mentioning fp8 / float8 / float-quantized) needs compute capability ≥ 8.9 (Ada+); Ampere dropped automatically.

TP=1 policies prefer unshared GPUs first, then co-locate. Alive Ray nodes only.

### Repository

- `ray_hive/` — planner, registry, deployment service, router, client API
- `manifests/` — KubeRay cluster, worker image, VRAM monitor
- `examples/` — deploy/inference experiments (`.env` / `.env.example`)
- `examples/requirements.txt` — example-only deps
- `examples/media/` — multimodal fixtures
- `basic_ray_tests/` — cluster / resource checks

Related: [rayify](https://github.com/BasicOverflow/rayify), a tool for converting scripts into Ray jobs.
