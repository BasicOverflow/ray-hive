# Ray K3s Deployment

Infrastructure as Code for deploying a production-ready Ray cluster on k3s using KubeRay. Supports vLLM inference serving, REST API job submission, and dynamic VRAM-based GPU scheduling.

## Overview

Deploy a Ray cluster on k3s with VRAM-based GPU scheduling for vLLM inference serving. Features dynamic GPU allocation, automatic replica placement, and heterogeneous GPU support.

## Architecture

```
Remote Devices/Scripts
    ↓
MetalLB LoadBalancer (Ray Dashboard + REST API)
    ↓
Ray Head Node (1 pod)
    ├── Ray Dashboard (port 8265)
    └── Ray REST API (/api/jobs)
    ↓
Ray Worker Pods
    ├── CPU Workers (on CPU-only nodes)
    └── GPU Workers (on GPU nodes)
        ├── All GPUs allocated per pod (dynamic)
        ├── VRAM DaemonSet → gpu_registry actor (per-GPU free/total + PyCUDA specs)
        └── vLLM replicas scheduled by VRAM availability
```

## Key Features

- **VRAM-Aware Scheduling**: Dynamic VRAM tracking via DaemonSet, global `gpu_registry` actor, exact VRAM requirements
- **vLLM Model Deployment**: Deploy via Ray Serve with VRAM reservation, multiple models per GPU, zero OOM guarantees
- **Automatic Placement**: Ray Serve places replicas based on available VRAM

## Repository Structure

- `manifests/` - Kubernetes manifests (KubeRay operator, Ray cluster, VRAM monitoring)
- `ray_hive/` - Python module (hive client, inference, core components)
- `examples/` - Example scripts
- `basic_ray_tests/` - Cluster testing scripts

## Quick Start

### Deploy Ray Cluster

```bash
# Deploy KubeRay operator (if not already installed)
kubectl apply -f manifests/kuberay-operator.yaml

# Deploy NVIDIA device plugin (if not already installed)
kubectl apply -f manifests/nvidia-device-plugin.yaml

# Deploy Ray cluster
kubectl apply -f manifests/raycluster.yaml

# Deploy vLLM install script + VRAM monitor DaemonSet
kubectl apply -f manifests/vllm-install-configmap.yaml
kubectl apply -f manifests/ray-vram-monitor-daemonset.yaml
```

### Install Ray Hive Module

```bash
# Install from local source
pip install -e .

# Or install from GitLab (update URL with your project)
pip install ray-hive --extra-index-url https://gitlab.com/api/v4/projects/.../packages/pypi/simple
```

### Deploy Models

**Using the Ray Hive Module:**

```python
from ray_hive import RayHive

scheduler = RayHive()

# Deploy model — planner auto-computes VRAM settings from HF config
# replicas=-1 deploys to all eligible GPUs
scheduler.deploy_model(
    model_id="qwen",
    model_name="Qwen/Qwen3-0.6B-GPTQ-Int8",
    max_input_prompt_length=1024,
    max_output_prompt_length=2048,
    replicas=6,
    enforce_eager=True,
    kv_cache_dtype="fp8",
)

# Single GPU pin
scheduler.deploy_model(
    model_id="qwen-test",
    model_name="Qwen/Qwen3-0.6B-GPTQ-Int8",
    max_input_prompt_length=1024,
    max_output_prompt_length=2048,
    gpu="ergos-06-nv:gpu0",
)

# Display VRAM state
scheduler.display_vram_state()
```

### Run Inference

**Using Standalone Inference Functions:**

```python
from ray_hive.inference import inference, a_inference, inference_batch

# Synchronous inference
result = inference("Hello!", model_id="my-model")

# Async inference
result = await a_inference("Hello!", model_id="my-model")

# Batch inference
results = inference_batch(
    ["Prompt 1", "Prompt 2", "Prompt 3"],
    model_id="my-model"
)

# Structured output
from pydantic import BaseModel

class Response(BaseModel):
    answer: str
    confidence: float

result = inference(
    "What is 2+2?",
    model_id="my-model",
    structured_output=Response
)

# Pass any vLLM SamplingParams kwargs (temperature, top_k, top_p, etc.)
result = inference("Hello!", model_id="my-model", temperature=0.7, top_k=50)
```

## How It Works

**GPU Registry**: Global singleton actor (`gpu_registry`) tracks live VRAM per GPU and deployment reservations.

**VRAM Monitoring**: DaemonSet on each GPU node calls `gpu_registry.update_gpu*` directly via Ray (no ConfigMap). PyCUDA specs once at startup; nvidia-smi VRAM every 0.5s. Start `RayHive()` first so the actor exists.

**Model Deployment**: `DeployService` singleton serializes deploys. Per GPU: `plan_deployment()` computes vLLM settings, `RayLLMActor` loads the model (pinned via custom Ray resources), `ModelRouter` exposes OpenAI-compatible `/v1` endpoints and least-queue routing.

**OpenAI HTTP API**: Each deployed model is reachable at `/{model_id}/v1/models`, `/{model_id}/v1/chat/completions`, `/{model_id}/v1/completions`.

## How the Router Works

The `ModelRouter` provides dynamic, capacity-aware load balancing across heterogeneous GPU deployments:

- **Least-Queue Routing**: Router tracks local queue depth per replica and routes to the replica with the lowest load relative to `max_num_seqs`.

- **OpenAI-Compatible Ingress**: FastAPI endpoints on the router deployment for `/v1/models`, `/v1/chat/completions`, `/v1/completions` (text-only for now; multimodal schema present but returns 400).

- **Heterogeneous GPU Support**: Per-GPU deployment plans computed independently from live VRAM and registry reservations.

## Manifests

- `kuberay-operator.yaml` - KubeRay operator
- `nvidia-device-plugin.yaml` - NVIDIA device plugin
- `raycluster.yaml` - Ray cluster deployment
- `vllm-install-configmap.yaml` - Shared vLLM install script for GPU worker init containers
- `ray-vram-monitor-daemonset.yaml` - Per-node VRAM + PyCUDA reporter (calls `gpu_registry` actor)

## Troubleshooting

Transient memory errors during initialization are expected when multiple replicas share a GPU. Ray Serve automatically retries failed deployments. If models consistently fail, verify VRAM requirements and ensure total VRAM doesn't exceed available GPU memory.

## API

### RayHive

- `deploy_model(model_id, model_name, max_input_prompt_length, max_output_prompt_length, replicas=-1, gpu=None, vram_weights_gb=None, max_num_batched_tokens=None, swap_space_per_instance=0, **vllm_kwargs)` — Deploy a model. `replicas=-1` uses all eligible GPUs. Optional overrides for weights and batched tokens. Any vLLM `LLM()` kwargs via `**vllm_kwargs`.
- `shutdown(model_id=None)` - Shutdown models (None = all)
- `get_vram_state()` - Get VRAM state dict
- `display_vram_state()` - Display VRAM state

### Inference Functions

- `inference(prompt, model_id, structured_output=None, max_tokens=None, **kwargs)` - Synchronous inference
- `a_inference(prompt, model_id, structured_output=None, max_tokens=None, **kwargs)` - Async inference
- `inference_batch(prompts, model_id, structured_output=None, max_tokens=None, **kwargs)` - Batch inference
- `a_inference_batch(prompts, model_id, structured_output=None, max_tokens=None, **kwargs)` - Async batch inference

All inference functions auto-discover deployments, support structured output (Pydantic), and accept any vLLM `SamplingParams` kwargs via `**kwargs` (e.g., `temperature`, `top_k`, `top_p`, `presence_penalty`, etc.). These can be changed per-request without redeploying.


## Future Enhancements

- **LangChain Compatibility**: LangChain LLM wrapper (TODO)
- **Vision/Audio Support**: Multimodal routing (schema ready, inference TODO)
- **Streaming**: Full streaming support (TODO)

## Related Repositories

- [rayify](https://github.com/BasicOverflow/rayify) - Ray script conversion tool

