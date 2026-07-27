# Ray Hive

Ray Hive is a small control layer for running a heterogeneous Ray cluster on k3s and placing vLLM model replicas according to live GPU memory.

The cluster remains a normal Ray cluster: CPU workers can run general distributed tasks while GPU workers host model deployments through Ray Serve.

## Contents

- [Architecture](#architecture)
- [Capabilities](#capabilities)
- [Model Planning](#model-planning)
- [GPU Monitoring and Placement](#gpu-monitoring-and-placement)
- [GPU Allocation Policies](#gpu-allocation-policies)
- [Tensor Parallelism](#tensor-parallelism)
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

- **Throughput-first planning:** By default, the planner uses the available VRAM budget to find the highest practical `max_num_seqs` and batched-token limit, maximizing concurrency while retaining runtime headroom (`gpu_budget_frac`: 0.90 single-GPU, 0.80 TP) for CUDA-graph capture and sampler scratch outside the util pool.
- **Heterogeneous deployment:** One model can be replicated across GPUs with different capacities. Each replica receives its own memory and concurrency plan so every GPU contributes as much throughput as it can.
- **Unified routing:** A model-level router ties those replicas together and sends work to the least-loaded GPU relative to its planned capacity.
- **Live VRAM scheduling:** Placement uses current GPU memory rather than static card specifications and reserves memory during deployment.
- **GPU sharing:** Multiple models can share a GPU when the registry and planner determine that enough VRAM remains. Auto-placement prefers GPUs with no hive reservations and only co-locates when no unshared eligible GPU remains; intentional sharing uses the same `gpu=` pin on two deploys with constrained `max_num_seqs` (see `examples/6_shared_gpu.py`).
- **Flexible placement:** Models can target specific GPUs, a requested replica count, or every eligible GPU.
- **Same-node tensor parallelism:** If a model does not fit on any one GPU, auto placement escalates to same-node TP; or pin an explicit `gpu=[...]` set. Each TP group is one Serve replica with vLLM’s multiprocessing backend.
- **Host RAM extension (`cpu_ram_per_instance`):** Sole hive arg for extending beyond GPU VRAM on **TP=1** replicas (weight spill only). If weights do not fit the GPU budget, overflow spills to host (`cpu_offload_gb`). KV offloading is not supported yet — passing `kv_offloading_size`, `kv_offloading_backend`, or `kv_transfer_config` raises. Do not pass `cpu_offload_gb` as a vLLM kwarg; use `cpu_ram_per_instance`. Not supported when TP>1 — use `cpu_ram_per_instance=0` and fit on GPU VRAM only.
- **General compute:** The same cluster continues to run normal distributed Ray tasks on its CPU workers (and GPU workers if not busy serving models).

## Model Planning

Before loading a model, Ray Hive reads its Hugging Face configuration and builds a per-GPU memory plan:

- **Weights** are estimated from the architecture, parameter dimensions, and model dtype (divided by TP size for per-GPU planning).
- **Attention/KV cache** is estimated from attention layers, KV heads, head size, context length, cache dtype, and concurrent sequences. Specialized attention layouts can provide their own calculation; TP sharding is applied in the planner, not in attention classes.
- **Runtime memory** includes a small system allowance (plus a coarse NCCL/TP term when TP>1), model-specific overhead, and activation memory based on batched tokens.
- **Host RAM (`cpu_ram_per_instance`, TP=1 only):** Weight spill only. If weights fit in the GPU budget, they stay on GPU and host budget is unused (`-1` resolves to 0). If not, overflow spills to host (auto `-1` sizes to that need, capped at 70% of Ray free host memory / replicas on host; `>0` is a hard ceiling). Concurrency (`max_num_seqs`) is planned from **GPU KV only**. Weight spill can free GPU VRAM and thereby raise concurrency. Activations, batched tokens, VRAM reservation, and `gpu_memory_utilization` stay GPU-only. TP>1 groups must fit on GPU VRAM with `cpu_ram_per_instance=0`. KV offload vLLM kwargs are rejected (not supported yet).

Each GPU (or TP group bottleneck) is planned independently, allowing different cards and existing deployments to have different concurrency limits.

## GPU Monitoring and Placement

A DaemonSet runs on every GPU node and reports live free/total VRAM and device information to a shared Ray actor. The registry combines those readings with pending and active model reservations so simultaneous deployments do not plan against the same memory.

Placement supports:

- `gpu="host:gpu0"` — pin one replica to one GPU.
- `gpu=[a,b,...]` with `replicas == len(list)` and `replicas > 1` — **N single-GPU pins** (one replica per listed GPU, TP=1 each).
- `gpu=[a,b,...]` with `replicas == 1` — **one same-node TP group** (`tp_size = len(list)`).
- With `gpu=None`, try single-GPU placement via `allocation_cls` (default: performance ranking); if nothing fits and `cpu_ram_per_instance=0`, escalate to same-node TP packs.
- Use `replicas=-1` to deploy on every eligible GPU (or every non-overlapping TP group) under that policy.

After placement, requests go to the least-loaded replica relative to its planned capacity.

## GPU Allocation Policies

When `gpu=` is not set, deploy picks GPUs through an `allocation_cls`. Abstract policy math lives in `ray_hive.core.gpu_alloc`; Ray helpers live in `ray_hive.core.ray_utils` / `ray_gpu_alloc`.

If `gpu=` or `gpu=[...]` is set, allocation policies are skipped and those pins are used (spill-aware VRAM fit via `cpu_ram_per_instance` applies on TP=1 only).

Policies implemented so far:

- **Performance** (`RayPerformanceAllocator`) — rank eligible GPUs by compute proxy; select top-N for replicas.
- **Conserve TDP** (`RayConserveTdpAllocator`) — rank eligible GPUs toward lower approximate TDP; SM count as tie-break.
- **FP8** (`RayFp8Allocator`) — same ranking as PerformanceAllocator; prefer Ada GPUs when FP8 is used.
- **Tensor parallel** (`RayTensorParallelAllocator`) — pack same-node GPU sets (usable ≈ N × min available); used automatically when single-GPU placement fails and `gpu=` is unset.

TP=1 policies (Performance / Conserve TDP / FP8) use a two-tier select: unshared GPUs (no pending/active hive reservations) first, ranked by policy score; only then partially occupied GPUs. Co-location is a last resort when unshared capacity is exhausted. Tensor-parallel select is unchanged (multi-GPU groups for one model).

Ray-wired allocators also drop GPUs whose host is not an Alive Ray node.

## Tensor Parallelism

Same-node only for now. Cross-node TP / pipeline parallel are not wired yet.

- **Auto (`gpu=None`):** try TP=1 (with optional `cpu_ram_per_instance` spill); if no GPU can hold weights+overhead and `cpu_ram_per_instance=0`, escalate TP=2,3,… via `RayTensorParallelAllocator` until a same-node pack fits on GPU VRAM only.
- **Pin TP (`gpu=[a,b,...]`, `replicas=1`):** that list *is* one TP group (`tp_size = len(list)`); must be same host.
- **Multi-pin (`gpu=[a,b,...]`, `replicas=len(list)`):** N independent TP=1 replicas, one per listed GPU (hosts may differ).
- **Pin (`gpu="host:gpu0"`):** single-GPU placement.
- **One TP replica per deploy:** a TP-sized `gpu=` list with `replicas` other than `1` or `len(gpu)` is rejected. To run a second TP copy, call `deploy_model` again with a different `model_id` / pin.
- `replicas` — number of independent GPUs when TP=1 (including multi-pin); number of independent TP *groups* when auto TP>1; must be `1` for a pinned TP group.
- Engine uses `distributed_executor_backend="mp"` (vLLM multiprocessing inside the Serve actor). Do not pass vLLM’s Ray executor for this path.
- Custom `attention_cls` stays architecture-only; the planner divides weights/KV by TP size.
- There is no user-facing `tensor_parallel_size` deploy arg — TP size comes from auto escalation or `len(gpu)` when `replicas=1`.

```python
# Force TP across two cards on one host
hive.deploy_model(
    model_id="qwen-tp",
    model_name="Qwen/Qwen3-8B-FP8",
    max_input_prompt_length=1024,
    max_output_prompt_length=2048,
    replicas=1,
    gpu=["ergos-02-nv:gpu0", "ergos-02-nv:gpu1"],
    vllm_kwargs={
        "trust_remote_code": True,
        "reasoning_parser": "qwen3",
        "default_chat_template_kwargs": {"enable_thinking": False},
    },
)

# Or omit gpu= — single GPU if it fits, else same-node TP
hive.deploy_model(
    model_id="qwen-auto",
    model_name="Qwen/Qwen3-8B-FP8",
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

## Basic Model Usage

```python
from ray_hive import RayHive
from ray_hive.inference import inference, inference_batch
from ray_hive.core.ray_gpu_alloc import RayPerformanceAllocator
# from ray_hive.core.model_specs import BaseAttentionSpecs  # subclass for custom KV sizing

hive = RayHive(address="ray://YOUR_RAY_HEAD_IP:10001")
status = hive.deploy_model(
    model_id="qwen",
    model_name="Qwen/Qwen3-0.6B-FP8",
    max_input_prompt_length=1024,
    max_output_prompt_length=2048,
    replicas=-1,
    allocation_cls=RayPerformanceAllocator,  # default; can omit
    # attention_cls=MyAttentionSpecs,  # omit to default to BaseAttentionSpecs (standard attention)
    vllm_kwargs={
        # planner overrides are lifted automatically from this dict
        # "max_num_seqs": 32,
        # "max_num_batched_tokens": 1024,
        # Qwen3-0.6B-FP8 suggested vLLM args (HF model card / Qwen deploy docs)
        "trust_remote_code": True,
        "reasoning_parser": "qwen3",  # model card also lists deepseek_r1 + deprecated enable-reasoning
        "default_chat_template_kwargs": {"enable_thinking": False},
    },
)
# status == {"model_id": "qwen", "status": "ready", "route": "/qwen", "replicas": {...}}

answer = inference("Explain Ray in one sentence.", model_id="qwen")
answers = inference_batch(["Prompt one", "Prompt two"], model_id="qwen")
```

Ray Hive-specific `deploy_model` arguments:

- `model_id` — local name used for the deployment, router, and API path.
- `model_name` — Hugging Face model ID or model path passed to vLLM.
- `max_input_prompt_length` / `max_output_prompt_length` — expected limits used to plan context memory and concurrency.
- `replicas` — number of GPUs when TP=1 (including multi-pin); number of TP *groups* when auto TP>1; must be `1` for a pinned TP group; `-1` uses every eligible GPU/group.
- `gpu` — optional pin: a string for one GPU; a list with `replicas=1` for one same-node TP group; a list with `replicas=len(list)` for N single-GPU pins. Overrides `allocation_cls`. Omit to auto-place (single GPU, else same-node TP).
- `cpu_ram_per_instance` — hive host-RAM extension arg (not a vLLM kwarg; default `0`; **TP=1 only**):
  - `0` — off (GPU-only; no weight spill)
  - `-1` — auto: budget exactly the weight spill this replica needs (0 if weights fit), capped at 70% of Ray free host memory / replicas on host
  - `>0` — hard host ceiling in GiB for weight spill (unused if weights fit on GPU)
  Policy: keep weights on GPU if they fit; else spill overflow to host. KV offloading is not supported yet. With TP>1, must be `0` (GPU VRAM only).
  Do not pass as vLLM kwargs: `kv_offloading_size`, `kv_offloading_backend`, `kv_transfer_config` (raises — KV offload unsupported), or `cpu_offload_gb` (raises — use `cpu_ram_per_instance`).
- `attention_cls` — optional `BaseAttentionSpecs` subclass for KV planning; defaults to standard attention (no TP awareness required).
- `allocation_cls` — optional single-GPU placement policy; defaults to `RayPerformanceAllocator` (ignored when `gpu=` is set; auto TP always uses `RayTensorParallelAllocator`).
- `idle_timeout` — seconds of inference inactivity before the model self-shutdowns; `-1` (default) means never, must be `-1` or a positive integer. Survives client script exit; uses the same cleanup as `hive.shutdown(model_id)`.
- `sleep_timeout` — seconds of inference inactivity before all replicas enter vLLM sleep (level 1: weights offloaded to CPU, KV discarded); `-1` (default) means never. Holds hive VRAM reservations while sleeping; next inference wakes replicas. When set, hive injects `enable_sleep_mode=True` into the engine and plans ~one extra weight copy of CuMemAllocator peak so KV still fits. Survives client script exit.
- When both `sleep_timeout` and `idle_timeout` are set, `idle_timeout` must be greater than `sleep_timeout` (sleep first, then full destroy after longer quiet).
- `vllm_kwargs` — dict forwarded to vLLM's `LLM(...)` constructor. Planner keys inside it (`max_num_seqs`, `max_num_batched_tokens`) are lifted into the deploy plan automatically and not double-applied to the engine. Serve-only keys like `default_chat_template_kwargs` are handled by the router.

`deploy_model` returns when the model is ready (router deployed and warmed):

```python
{"model_id": "...", "status": "ready", "route": "/...", "replicas": {...}}
```


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
- `examples/` — model deployment and inference experiments (`.env` / `.env.example` for cluster URLs; also used by `basic_ray_tests/`)
- `basic_ray_tests/` — general cluster and resource checks

Related: [rayify](https://github.com/BasicOverflow/rayify), a tool for converting scripts into Ray jobs.