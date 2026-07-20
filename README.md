# Ray Hive

Ray Hive is a small control layer for running a heterogeneous Ray cluster on k3s and placing vLLM model replicas according to live GPU memory.

The cluster remains a normal Ray cluster: CPU workers can run general distributed tasks while GPU workers host model deployments through Ray Serve.

## Contents

- [Architecture](#architecture)
- [Capabilities](#capabilities)
- [Model Planning](#model-planning)
- [GPU Monitoring and Placement](#gpu-monitoring-and-placement)
- [GPU Allocation Policies](#gpu-allocation-policies)
- [Basic Model Usage](#basic-model-usage)
- [General Ray CPU Tasks](#general-ray-cpu-tasks)
- [Repository](#repository)

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

- **Throughput-first planning:** By default, the planner uses the available VRAM budget to find the highest practical `max_num_seqs` and batched-token limit, maximizing concurrency while retaining 5% runtime headroom to avoid OOM crashes.
- **Heterogeneous deployment:** One model can be replicated across GPUs with different capacities. Each replica receives its own memory and concurrency plan so every GPU contributes as much throughput as it can.
- **Unified routing:** A model-level router ties those replicas together and sends work to the least-loaded GPU relative to its planned capacity.
- **Live VRAM scheduling:** Placement uses current GPU memory rather than static card specifications and reserves memory during deployment.
- **GPU sharing:** Multiple models can share a GPU when the registry and planner determine that enough VRAM remains.
- **Flexible placement:** Models can target specific GPUs, a requested replica count, or every eligible GPU.
- **General compute:** The same cluster continues to run normal distributed Ray tasks on its CPU workers (and GPU workers if not busy serving models).

## Model Planning

Before loading a model, Ray Hive reads its Hugging Face configuration and builds a per-GPU memory plan:

- **Weights** are estimated from the architecture, parameter dimensions, and model dtype.
- **Attention/KV cache** is estimated from attention layers, KV heads, head size, context length, cache dtype, and concurrent sequences. Specialized attention layouts can provide their own calculation.
- **Runtime memory** includes a small system allowance, model-specific overhead, and activation memory based on batched tokens.
- **Concurrency** (`max_num_seqs`) and batched-token limits are pushed as high as the remaining VRAM allows, with 5% left outside the plan for runtime headroom.

Each GPU is planned independently, allowing different cards and existing deployments to have different concurrency limits.

## GPU Monitoring and Placement

A DaemonSet runs on every GPU node and reports live free/total VRAM and device information to a shared Ray actor. The registry combines those readings with pending and active model reservations so simultaneous deployments do not plan against the same memory.

Placement supports:

- Pin a replica to one GPU or an explicit list of GPUs (`gpu=` — skips allocation policies).
- Use an allocation policy to pick the top `replicas` eligible GPUs (default: performance ranking).
- Use `replicas=-1` to deploy on every eligible GPU under that policy.

After placement, requests go to the replica with the lowest queue depth relative to that replica's planned concurrency.

## GPU Allocation Policies

When `gpu=` is not set, deploy picks GPUs through an `allocation_cls`. Abstract policy math lives in `ray_hive.core.gpu_alloc`; `ray_hive.core.ray_utils` / `ray_gpu_alloc`.

If `gpu=` or `gpu=[...]` is set, allocation policies are skipped and those pins are used (VRAM checks only).

Policies implemented so far:

- **Performance** (`RayPerformanceAllocator`) — rank eligible GPUs by compute proxy; select top-N for replicas.
- **Conserve TDP** (`RayConserveTdpAllocator`) — rank eligible GPUs toward lower approximate TDP; SM count as tie-break.
- **FP8** (`RayFp8Allocator`) — same ranking as PerformanceAllocator; prefer Ada GPUs when FP8 is used.
- **Tensor parallel** (`TensorParallelAllocator`) — placeholder for future symmetric TP set selection.

Ray-wired allocators also drop GPUs whose host is not an Alive Ray node.

## Basic Model Usage

```python
from ray_hive import RayHive
from ray_hive.inference import inference, inference_batch
from ray_hive.core.ray_gpu_alloc import RayPerformanceAllocator
# from ray_hive.core.model_specs import BaseAttentionSpecs  # subclass for custom KV sizing

hive = RayHive(address="ray://YOUR_RAY_HEAD_IP:10001")
hive.deploy_model(
    model_id="qwen",
    model_name="Qwen/Qwen3-0.6B-FP8",
    max_input_prompt_length=1024,
    max_output_prompt_length=2048,
    replicas=-1,
    allocation_cls=RayPerformanceAllocator,  # default; can omit
    # attention_cls=MyAttentionSpecs,  # omit to default to BaseAttentionSpecs (standard attention)
    # Qwen3-0.6B-FP8 suggested vLLM args (HF model card / Qwen deploy docs)
    trust_remote_code=True,
    reasoning_parser="qwen3",  # model card also lists deepseek_r1 + deprecated enable-reasoning
    default_chat_template_kwargs={"enable_thinking": False},
)

answer = inference("Explain Ray in one sentence.", model_id="qwen")
answers = inference_batch(["Prompt one", "Prompt two"], model_id="qwen")
```

Ray Hive-specific `deploy_model` arguments:

- `model_id` — local name used for the deployment, router, and API path.
- `model_name` — Hugging Face model ID or model path passed to vLLM.
- `max_input_prompt_length` / `max_output_prompt_length` — expected limits used to plan context memory and concurrency.
- `replicas` — number of GPUs to deploy on; `-1` uses every eligible GPU.
- `gpu` — optional GPU key or list of keys (for example, `ergos-06-nv:gpu0`) for explicit placement; overrides `allocation_cls`.
- `max_num_seqs` / `max_num_batched_tokens` — optional overrides for the planner's estimates.
- `swap_space_per_instance` — CPU swap space in GiB available to each vLLM instance.
- `attention_cls` — optional `BaseAttentionSpecs` subclass for KV planning; defaults to standard attention.
- `allocation_cls` — optional GPU placement policy; defaults to `RayPerformanceAllocator` (ignored when `gpu=` is set).
- `idle_timeout` — seconds of inference inactivity before the model self-shutdowns; `-1` (default) means never, must be `-1` or a positive integer. Survives client script exit; uses the same cleanup as `hive.shutdown(model_id)`.

Any additional keyword arguments are forwarded to vLLM's `LLM(...)` constructor.

Structured output accepts a Pydantic model:

```python
from pydantic import BaseModel

class Answer(BaseModel):
    summary: str
    confidence: float

result = inference("Summarize Ray.", model_id="qwen", structured_output=Answer)
```

Each deployed model also exposes an OpenAI-compatible HTTP API on Ray Serve (default port `8000`):

```bash
curl http://YOUR_RAY_HEAD_IP:8000/qwen/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Explain Ray briefly."}]}'
```

The current inference helper and HTTP router support text generation. The deployment and planning layers are intended to cover vLLM-compatible:

- text-generation models
- image/multimodal models
- audio models
- embedding models

Image, audio, and embedding request/response routing still require their model-specific interfaces.

## General Ray CPU Tasks

```python
import ray

ray.init(address="ray://YOUR_RAY_HEAD_IP:10001")

@ray.remote
def square(value):
    return value * value

results = ray.get([square.remote(value) for value in range(10)])
```

## Repository

- `ray_hive/` — planner, registry, deployment service, router, and client API
- `manifests/` — KubeRay cluster, worker image, and VRAM monitor definitions
- `examples/` — model deployment and inference experiments (use `.env` / `.env.example` for cluster URLs)
- `basic_ray_tests/` — general cluster and resource checks

Related: [rayify](https://github.com/BasicOverflow/rayify), a tool for converting scripts into Ray jobs.