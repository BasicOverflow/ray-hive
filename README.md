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
        ├── VRAM reported as custom resource
        └── vLLM replicas scheduled by VRAM availability
```

## Key Features

- **VRAM-Aware Scheduling**: Dynamic VRAM tracking via DaemonSet, global allocator actor, exact VRAM requirements
- **vLLM Model Deployment**: Deploy via Ray Serve with VRAM reservation, multiple models per GPU, zero OOM guarantees
- **Automatic Placement**: Ray Serve places replicas based on available VRAM

## Repository Structure

- `manifests/` - Kubernetes manifests (KubeRay operator, Ray cluster, VRAM monitoring)
- `ray_hive/` - Python module (client, inference, core components)
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

# Deploy VRAM monitor
kubectl apply -f manifests/vram-scheduler-configmap.yaml
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

# Deploy model with specific number of replicas
scheduler.deploy_model(
    model_id="qwen",
    model_name="Qwen/Qwen3-0.6B-GPTQ-Int8",
    vram_weights_gb=0.763,  # Model weights only (KV cache calculated separately)
    max_input_prompt_length=1024,  # Maximum input prompt length
    max_output_prompt_length=2048,  # Maximum output tokens
    max_num_seqs=850,  # Maximum concurrent sequences (required)
    max_num_batched_tokens=16384,  # Maximum batched tokens (required)
    replicas=6,
    gpu_utilization_target=0.96,  # GPU VRAM utilization target (default 0.96, can be overridden)
    enforce_eager=True,
    kv_cache_dtype="fp8"
)

# Deploy in test mode (single replica on GPU with most VRAM)
scheduler.deploy_model(
    model_id="qwen-test",
    model_name="Qwen/Qwen3-0.6B-GPTQ-Int8",
    vram_weights_gb=0.763,
    max_input_prompt_length=1024,
    max_output_prompt_length=2048,
    max_num_seqs=850,
    max_num_batched_tokens=16384,
    gpu="ergos-06-nv:gpu0"  # Optional: specify GPU(s) to deploy to
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
```

## How It Works

**VRAM Allocator**: Global singleton actor tracks VRAM state per GPU and manages reservations.

**VRAM Monitoring**: DaemonSet monitors VRAM on each GPU node via `nvidia-smi` every 0.5s.

**Model Deployment**: `RayHive.deploy_model()` queries available GPUs, creates one `VLLMModel` deployment per GPU (targeted via `CUDA_VISIBLE_DEVICES`), and sets up `ModelRouter` for load balancing.

**VRAM Reservation**: Each replica reserves VRAM before loading, uses fractional GPU allocation, and hard-limits memory usage to prevent OOM.

## How the Router Works

The `ModelRouter` provides dynamic, capacity-aware load balancing across heterogeneous GPU deployments:

- **Dynamic Capacity-Aware Routing**: Router tracks real-time queue depth and available capacity per GPU deployment, routing requests to GPUs with the most available capacity.

- **Performance-Based Selection**: Routes requests to GPUs based on performance factors (SM count, VRAM) and current load. Faster GPUs receive more work, preventing slower GPUs from blocking the pipeline.

- **Request Size Estimation**: Analyzes incoming requests (prompt length, max_tokens) to match them with appropriate GPU capacity. Large requests are routed to higher-capacity GPUs.

- **Prevents Stalling**: Faster GPUs get more work, slower GPUs don't block the pipeline. The router continuously adapts to changing load conditions.

- **Heterogeneous GPU Support**: Automatically adapts to different GPU types and capacities in the cluster. Performance factors are calculated dynamically based on detected maximum VRAM across all deployments.

## Manifests

- `kuberay-operator.yaml` - KubeRay operator
- `nvidia-device-plugin.yaml` - NVIDIA device plugin
- `raycluster.yaml` - Ray cluster deployment
- `ray-vram-monitor-daemonset.yaml` - VRAM monitoring
- `vram-scheduler-configmap.yaml` - VRAM scheduler scripts

## Troubleshooting

Transient memory errors during initialization are expected when multiple replicas share a GPU. Ray Serve automatically retries failed deployments. If models consistently fail, verify VRAM requirements and ensure total VRAM doesn't exceed available GPU memory.

## API

### RayHive

- `deploy_model(model_id, model_name, vram_weights_gb, max_input_prompt_length, max_output_prompt_length, max_num_seqs, max_num_batched_tokens, replicas=None, gpu=None, gpu_utilization_target=0.96, swap_space_per_instance=0, **vllm_kwargs)` - Deploy a model. `replicas` can be an integer, `"max"` to deploy to all available GPUs, or `None` to use all available GPUs. `gpu` can be a string (single GPU), a list of strings (must match num_replicas length), or `None` to let library determine placement. `swap_space_per_instance` is the swap space per instance in GB. `gpu_utilization_target` controls VRAM budget calculation (default 0.96) and can be overridden. `gpu_memory_utilization` can also be set via `vllm_kwargs` to override the value passed to vLLM.
- `shutdown(model_id=None)` - Shutdown models (None = all)
- `get_vram_state()` - Get VRAM state dict
- `display_vram_state()` - Display VRAM state

### Inference Functions

- `inference(prompt, model_id, structured_output=None, max_tokens=None, ...)` - Synchronous inference
- `a_inference(prompt, model_id, ...)` - Async inference
- `inference_batch(prompts, model_id, batch_size=None, ...)` - Batch inference
- `a_inference_batch(prompts, model_id, batch_size=None, ...)` - Async batch inference

All inference functions auto-discover deployments, support structured output (Pydantic), and accept vLLM sampling parameters.


## Future Enhancements

- **LangChain Compatibility**: LangChain LLM wrapper (TODO)
- **Vision/Audio Support**: Support for vision and audio models (TODO)
- **OpenAI API Compatibility**: OpenAI-compatible endpoints (TODO)
- **Streaming**: Full streaming support (TODO)

## Related Repositories

- [rayify](https://github.com/BasicOverflow/rayify) - Ray script conversion tool

