# ray-hive

**ray-hive** is a shared-cluster toolkit for [Ray](https://www.ray.io/): it makes distributed Ray development simpler and more reliable by baking resilience into reusable primitives so each project ships less Ray glue and has a smoother path to production.

You run one heterogeneous Ray cluster (CPU workers for jobs, GPU workers for models). Multiple apps share that cluster — for example a **quotes-pipeline** (batch IO + Serve API), a **catalog-search** service (embeddings / retrieval), plus ad-hoc vLLM deploys and Train runs — without fighting over Serve routes, model ids, or cron windows. Hive owns the boring distributed concerns (placement envelopes, admission, durable queues, claim registry, Serve cool-down, VRAM-aware vLLM placement) so your project code stays domain-shaped.

Install extras match how you use the cluster:

```text
pip install "ray-hive"               # core: jobs + cluster + meta hooks
pip install "ray-hive[vllm]"         # + Ray Serve / vLLM path
pip install "ray-hive[train]"        # + Ray Train / custom infer
pip install "ray-hive[all]"          # vllm + train
```

On connect you get one stdout line: `ray-hive <ver> — https://github.com/BasicOverflow/ray-hive`.

**What you get**

| Layer | Role |
|-------|------|
| **vLLM Serve** (`[vllm]`) | Live-VRAM planning, heterogeneous GPU placement, sleep/idle, OpenAI HTTP |
| **Jobs primitives** (core) | IO/CPU envelopes, fan-out + admission, rate gates, durable queues, cron board |
| **Train** (`[train]`) | Custom PyTorch/HF loops with the same GPU claim model |
| **Meta-supervisor** | Cluster-wide claims, cron calendar, sticky model revive (LXC/systemd host) |

**Design intent:** cut Ray-related code per project. Resilience (reconnect, cool-down, head/vram-monitor avoidance, durable pending work while Ray is down, collision GC) lives in hive so prod systems spend less time rediscovering the same failure modes.

---

## Contents

- [vLLM Serve](#vllm-serve-optional-vllm)
- [Placement](#placement)
- [Allocation policies](#allocation-policies)
- [Custom attention](#custom-attention)
- [Multimodal and context](#multimodal-and-context)
- [Embeddings](#embeddings)
- [Lifecycle](#lifecycle)
- [OpenAI HTTP](#openai-http)
- [Jobs primitives](#jobs-primitives)
- [Meta-supervisor](#meta-supervisor)
- [Train / custom infer](#train--custom-infer-optional-train)
- [Ops](#ops)
- [Tests](#tests)

---

## vLLM Serve (optional `[vllm]`)

Ray Hive’s original job: run a heterogeneous Ray cluster and place **vLLM** model replicas according to **live GPU memory**. CPU workers keep running general Ray tasks; GPU workers host Serve replicas. KubeRay manages pods; hive adds planning, per-GPU placement, reservations, and routing.

```text
Clients
  └─ Ray head
      ├─ CPU workers → jobs / general Ray tasks
      └─ GPU workers → vLLM replicas
           ├─ VRAM monitor DaemonSet
           ├─ shared GPU registry
           └─ Ray Serve model routers
```

### Capabilities

- **Throughput-first planning** — max practical `max_num_seqs` / batched tokens within a VRAM budget (`gpu_budget_frac` 0.95). With `max_input_prompt_length="auto"` + fixed `max_num_seqs`, grow text context (floor 256) instead. At `max_num_seqs=1`, auto may shrink below 256 so weights + speculative draft still fit. Hybrid GDN/Mamba models cap auto input at 32768 so leftover VRAM is not spent on an unbounded attention window.
- **Heterogeneous replicas** — per-GPU plans so different cards each contribute what they can.
- **Least-loaded routing** — relative to each replica’s planned capacity.
- **Live VRAM scheduling** — registry + reservations; see `examples/4_test_allocation_policies.py`.
- **GPU sharing** — co-locate when needed; intentional share via same `gpu=` pin (`examples/6_shared_gpu.py`).
- **Flexible placement** — pin, N replicas, or `replicas=-1` (`examples/1_test_model_configs.py`).
- **Flagship deploy reference** — Qwen3.8-27B + Nemotron 3.5 Lightning variants (`examples/15_flagship_models.py`).
- **Same-node TP** — auto escalate or pin a GPU list (`examples/5_tensor_parallel.py`).
- **Custom attention** — subclass using HF config fields (`examples/3_custom_attention.py`).
- **Multimodal generate** — image / video / audio (`examples/8`–`9`, `12`–`14`).
- **Embeddings** — `runner="pooling"` (`examples/10`–`11`).
- **Sleep / idle** — level-1 sleep then optional self-destroy (`examples/7_sleep_idle_timeout.py`).
- **OpenAI HTTP** — per-model routes + cluster `/v1` gateway (`examples/2_test_inference.py`).

**Out of scope:** precomputed multimodal embeds as *input* (`enable_mm_embeds`).

### Quick start

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
  max_input_prompt_length  1024
  max_output_prompt_length 2048
  max_model_len            3072
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

Structured output:

```python
from pydantic import BaseModel

class Answer(BaseModel):
    summary: str
    confidence: float

result = inference("Summarize Ray.", model_id="qwen", structured_output=Answer)
```

Inside an asyncio loop use `a_inference` / `a_inference_batch` instead of the sync helpers.

---

## Placement

- `gpu="host:gpu0"` — one replica on one GPU.
- `gpu=[a,b,...]` + `replicas=len(list)` — N single-GPU pins (TP=1 each).
- `gpu=[a,b,...]` + `replicas=1` — one same-node TP group.
- `gpu=None` — auto place (single GPU via `allocation_cls`, default `RayPerformanceAllocator`; else same-node TP).
- `replicas=-1` — every eligible GPU / TP group.

### Auto input length

By default context lengths are a fixed contract and omitted `max_num_seqs` is packed to fill VRAM.

`max_input_prompt_length="auto"` does the inverse for **text input only**: you must pass `max_num_seqs` in `vllm_kwargs`, output length stays fixed, and the planner grows text input from a floor of **256** until the VRAM budget is filled (capped by HF `max_position_embeddings` / `model_max_length` when present). When `max_num_seqs=1` and even 256 does not leave KV room after weights + speculative draft, auto shrinks toward 1 token. Multimodal placeholder tokens are still added on top of the chosen text length, so `max_model_len` always covers MM + output.

`speculative_config` is part of the memory plan: in-checkpoint MTP reserves an extra compute-dtype `lm_head` (and `num_nextn_predict_layers` when set); a separate draft `model` adds that checkpoint’s weights and KV.

```python
hive.estimate_vram(
    "Qwen/Qwen3-0.6B-FP8",
    max_input_prompt_length="auto",
    max_output_prompt_length=512,
    replicas=1,
    vllm_kwargs={
        "max_num_seqs": 8,
        "trust_remote_code": True,
        "reasoning_parser": "qwen3",
        "default_chat_template_kwargs": {"enable_thinking": False},
    },
)
```

Passing `"auto"` without `max_num_seqs` raises `ConfigError`.

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

All eligible GPUs: `replicas=-1`. Same-node TP=2: pass a GPU list with `replicas=1`. Live registry: `hive.get_vram_state()`.

---

## Allocation policies

When `gpu=` is unset, `allocation_cls` picks GPUs (`ray_hive.core.ray_gpu_alloc`). Pins skip policy ranking but still check VRAM fit and arch taints. Default: `RayPerformanceAllocator`.

- `RayPerformanceAllocator` — rank by compute (SM count, light bandwidth tie-break); top-N.
- `RayConserveTdpAllocator` — prefer lower approx TDP; SM count tie-break.
- `RayTensorParallelAllocator` — same-node packs of size `tensor_parallel_size`; auto when single-GPU fails.

**Arch taint:** native FP8 needs compute capability ≥ 8.9 (Ada+); Ampere dropped automatically.

TP=1 policies prefer unshared GPUs first, then co-locate. Alive Ray nodes only. See `examples/4_test_allocation_policies.py` and `examples/5_tensor_parallel.py`.

---

## Custom attention

VRAM planning sizes the KV cache through an attention-specs class. Default `BaseAttentionSpecs` (or `MultimodalAttentionSpecs` for MM). Pass a subclass via `attention_cls` on `estimate_vram` / `deploy_model`.

```python
from ray_hive.core.model_specs import BaseAttentionSpecs

class MyAttention(BaseAttentionSpecs):
    @property
    def kv_heads(self) -> int:
        return self.hf_params["num_key_value_heads"]

hive.deploy_model(..., attention_cls=MyAttention)
```

See `examples/3_custom_attention.py`.

---

## Multimodal and context

**Text-only models** — pass a string (or text chat messages). No `limit_mm_per_prompt`.

**MM models** — planner uses `limit_mm_per_prompt` in `vllm_kwargs` to size worst-case image / video / audio placeholders. If omitted on an MM HF config, defaults come from `vision_config` / `audio_config`. Set counts explicitly to enable or disable modalities.

**Token budget** — `max_input_prompt_length` is the **text** side only. Effective input ≈ text + MM placeholders; `max_model_len ≈ effective_input + max_output_prompt_length` (use output `0` for pooling). If that cannot cover placeholders + output, planning raises `MmContextError`. With `"auto"` input, text grows from 256 while MM placeholders stay reserved.

**Requests vs planning** — on an MM deploy you can still send text-only strings; that does **not** shrink the VRAM plan. For text-only *planning* on an MM checkpoint, zero unused modalities (`{"image": 0, "video": 0, "audio": 0}`).

```python
from ray_hive.core.ray_utils import file_to_data_url
from ray_hive.inference import inference

hive.deploy_model(
    model_id="vl",
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    max_input_prompt_length=2048,
    max_output_prompt_length=256,
    replicas=1,
    vllm_kwargs={"trust_remote_code": True, "limit_mm_per_prompt": {"image": 1}},
)

messages = [{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": file_to_data_url("photo.png")}},
        {"type": "text", "text": "Describe this image."},
    ],
}]
answer = inference(messages, model_id="vl", max_tokens=64)
```

Content part types: `text`, `image_url`, `video_url` / `video`, `audio_url` / `input_audio`. Fixtures: `examples/media/`. Demos: `examples/8`–`9`, `12`–`14`.

---

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

```bash
curl http://YOUR_RAY_HEAD_IP:8000/embed/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"embed","input":["hello world"]}'
```

See `examples/10_text_embeddings.py` and `examples/11_multimodal_embeddings.py`.

---

## Lifecycle

Sleep after quiet, then optional full destroy (`idle_timeout` must be greater than `sleep_timeout` when both are set). While sleeping, the running footprint stays reserved so other deploys cannot take the space a wake needs. Pass `sleep_peak_factor` in `vllm_kwargs` only if you need an extra weight-sized hold.

```python
hive.deploy_model(
    model_id="qwen",
    model_name="Qwen/Qwen3-0.6B-FP8",
    max_input_prompt_length=512,
    max_output_prompt_length=256,
    replicas=1,
    sleep_timeout=60,
    idle_timeout=300,
    vllm_kwargs={
        "trust_remote_code": True,
        "reasoning_parser": "qwen3",
        "default_chat_template_kwargs": {"enable_thinking": False},
    },
)

hive.shutdown("qwen")   # one model
hive.shutdown()         # all *hive* models only (never serve.shutdown — other apps stay up)
free = hive.free_gpus(min_free_gb=8.0)
```

```python
from ray_hive import kill_gpu_registry
kill_gpu_registry()     # explicit rebuild only — not part of normal shutdown
```

See `examples/7_sleep_idle_timeout.py` and `examples/0_shutdown_models.py`. Sticky pin when `HIVE_SUPERVISOR_STATE` is set (meta-supervisor). GCS FT Redis: `manifests/ray-gcs-redis.yaml`.

---

## OpenAI HTTP

Per-model route and cluster-wide `/v1` gateway (port `8000`). Gateway starts on `deploy_model`; for an already-running cluster call `hive.ensure_openai_api()`.

```bash
curl http://YOUR_RAY_HEAD_IP:8000/qwen/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Explain Ray briefly."}]}'

# Open WebUI: API URL = http://YOUR_RAY_HEAD_IP:8000/v1
curl http://YOUR_RAY_HEAD_IP:8000/v1/models
```

Streaming, structured JSON, and LangChain: `examples/2_test_inference.py`.

---

## Jobs primitives

These are the building blocks for **non-LLM** Ray work (and for the CPU side of LLM apps). They exist so every project does not re-implement placement, backpressure, and crash recovery.

### ProjectContext + ClusterRegistry

**What:** a named claim book for the whole cluster.  
**Why:** two projects must not both own `/quotes`, the same `model_id`, or the same cron id.  
**Where:** detached actor in namespace `ray_hive_system`.

```python
from ray_hive.cluster import ProjectContext, init_ray_for_project

ctx = ProjectContext(
    project_id="quotes-pipeline",
    namespace="quotes-pipeline",
    serve_route_prefix="/quotes",
)
init_ray_for_project("ray://YOUR_RAY_HEAD_IP:10001", ctx)
```

Miss heartbeats long enough and the meta-supervisor GC marks the project dead and releases claims.

### Envelopes (`io_remote` / `cpu_remote` / `ops_remote`)

**What:** `@ray.remote` option packs with safe defaults.  
**Why:** accidental scheduling onto the Ray head or onto **vram-monitor** pods (memory-only, no CPU) silently fails or starves the head.  
**Rule:** workload envelopes always set `label_selector` so tasks stay on worker node-groups (`!headgroup`). No opt-out. Use `ops_remote` for head / every-node ops (Serve restart, disk prune).

```python
from ray_hive.jobs import io_remote, cpu_remote

@io_remote()
def fetch_chunk(urls: list[str]) -> list:
    ...

@cpu_remote(num_cpus=-1)  # -1 → AdmissionBroker suggests width
def stitch(batch: list) -> list:
    ...
```

### BatchFanout + AdmissionBroker + TaskShape

**What:** submit work in chunks with `ray.wait` backpressure; width comes from live cluster CPU/RAM, not a hardcoded `MAX_TASKS`.  
**Why:** fixed concurrency either underuses the cluster or OOMs it when another project is busy.

```python
from ray_hive.jobs import BatchFanout, TaskShape

fan = BatchFanout()
shape = TaskShape(batch_size=32, max_ceiling=64, cpus_per_task=0.001)
fan.run("ingest", items, lambda batch: fetch_chunk.remote(batch), shape)
```

`max_ceiling` / `max_in_flight` are safety rails. If the broker actor is down, fan-out still runs with those rails.

### RateGate

**What:** a named cluster actor that serializes / paces external HTTP or vendor APIs.  
**Why:** per-process sleeps do not coordinate across workers.

```python
from ray_hive.jobs import get_or_create_rate_gate
import ray

gate = get_or_create_rate_gate(
    "vendor-http-gate",
    namespace="quotes-pipeline",
    min_interval=2.5,
)
ray.get(gate.acquire.remote())
# call external API
ray.get(gate.record_success.remote())
```

### DurableJobQueue

**What:** SQLite-backed pending/running/done queue on the project supervisor host.  
**Why:** Ray (and the whole worker set) can be offline for hours; catch-up work must survive.

```python
from ray_hive.jobs import DurableJobQueue

q = DurableJobQueue("/var/lib/quotes-pipeline/queue.db")
q.enqueue("daily-bar-sync", {"day": "2024-06-01"})
job = q.claim_next(["daily-bar-sync"])
q.complete(job["id"])
```

Stale `running` rows requeue after `stale_running_s`.

### CronSchedule + cron board

**What:** project declares schedules; meta holds a cluster-wide board (`next_fire`, `load_class`).  
**Why:** utilization planning across projects (IO vs CPU vs GPU windows) without one supervisor owning everyone’s wall clock.

```python
from ray_hive.jobs import CronSchedule, get_cron_board
import ray

CronSchedule(
    "quotes-daily-bar-sync",
    "quotes-pipeline",
    "16:05",
    load_class="io",
)
board = get_cron_board()
print(ray.get(board.list_entries.remote()))
```

Each project’s supervisor still **fires** the work; the board is the shared calendar.

### JobProgress

**What:** durable “which ids are done” for long fan-outs (embeddings, backfills).  
**Why:** restart mid-job without redoing completed shards.

```python
from ray_hive.jobs import JobProgress

prog = JobProgress("/var/lib/catalog-search/progress.db", "embed-v1")
todo = prog.pending(all_ids)
for i in todo:
    ...
    prog.mark_done(i)
```

### ScratchShardStore / ServeAppHeal / NodeBroadcast / MapShuffleReduce

- **ScratchShardStore** — local scratch generations under a prefix; safe wipe of orphan trees.
- **ServeAppHeal** — deploy/replace a Serve app with cool-down + route claim (so two projects do not stomp `/quotes`).
- **NodeBroadcast** — one ops task per alive node (disk prune, cache flush).
- **MapShuffleReduce** — small map→shuffle→reduce helper for CPU pipelines.

Request coalescing (identical in-flight HTTP→remote) stays in your app, not in hive.

---

## Meta-supervisor

LXC/systemd entry: `python -m ray_hive.supervisor.main` (`deploy/`).

Keeps **ClusterRegistry**, **AdmissionBroker**, and **cron board** warm, GCs dead projects, and redeploys sticky **vLLM** models when claimed. It does **not** replace per-project supervisors (e.g. `quotes-pipeline` or `catalog-search`) — those register into it and own their domain queues / Serve apps.

State file: `HIVE_SUPERVISOR_STATE` (SQLite): sticky models, projects, cron_board mirror, queues, trains, claims.

```bash
bash deploy/install.sh
# journalctl -u ray-hive-supervisor -f   # on the supervisor host
```

Worker images must include `ray-hive` (CPU: `manifests/Dockerfile.ray-cpu-hive`; GPU: `Dockerfile.ray-vllm-worker`) so system actors created via Ray Client can import the package on the cluster.

---

## Train / custom infer (optional `[train]`)

Sibling to vLLM — your `nn.Module` / HF code, VRAM claims via the same registry.

```python
from ray_hive.train import TrainJob, gpu_remote

@gpu_remote()
def worker_step(batch):
    ...

job = TrainJob("run-1", "gpu-rack-a", train_fn=my_loop, num_workers=2)
job.run()
```

`CustomInferService` claims a `model_id` then calls your deploy/shutdown callables.

---

## Ops

**GCS FT (Redis):** `manifests/ray-gcs-redis.yaml` + `gcsFaultToleranceOptions` on the RayCluster. After head failover, Serve controllers can revive on surviving workers. Hive’s in-memory `gpu_registry` does **not** survive GCS alone — the supervisor recreates it.

```bash
kubectl apply -f manifests/ray-gcs-redis.yaml
kubectl apply -f manifests/kuberay-operator.yaml
kubectl apply -f manifests/raycluster.yaml
```

**Images**

| Role | Dockerfile | Example tag |
|------|------------|-------------|
| Head + CPU workers | `manifests/Dockerfile.ray-cpu-hive` | `…/ray-cpu-hive:2.56.0-0.2.1` |
| GPU workers | `manifests/Dockerfile.ray-vllm-worker` | `…/ray-vllm-worker:2.56.0-vllm028-ocr2-0.2.1` |

Gitea PyPI (LAN): `http://10.0.0.52:3000/api/packages/admin/pypi/simple/` — used by CT installs and as a runtime_env fallback when creating system actors.

**Repository layout**

- `ray_hive/` — vLLM planner/registry/deploy, jobs, cluster, train, supervisor
- `deploy/` — LXC/systemd install for the meta-supervisor
- `manifests/` — KubeRay cluster, GCS Redis FT, worker images, VRAM monitor
- `examples/` — deploy/inference experiments
- `tests/` — unit / Ray smoke / live / chaos

Related: [rayify](https://github.com/BasicOverflow/rayify) (script → Ray job packaging).

---

## Tests

```bash
pip install -e ".[dev]"

# Parallel unit suite (no Ray / no LLM)
pytest -n auto -m "not ray and not live"
```

`ray` / `live` require `RAY_ADDRESS`:

```bash
export RAY_ADDRESS=ray://YOUR_RAY_HEAD_IP:10001   # or PowerShell $env:RAY_ADDRESS=...
pytest -m ray    # cluster smoke, no LLM
pytest -m live   # Deploy cycles; needs GPUs + model download
```

Do **not** use `pytest -n` for live GPU claims. Markers: `ray`, `live`, `nightly`, `chaos`.
