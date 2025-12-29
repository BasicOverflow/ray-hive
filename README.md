# Ray K3s Deployment

Infrastructure as Code for deploying a production-ready Ray cluster on k3s using KubeRay. Supports vLLM inference serving, REST API job submission, and dynamic VRAM-based GPU scheduling.

## Overview

This repository contains all the manifests, scripts, and documentation needed to deploy a Ray cluster on k3s with:

- **KubeRay Operator**: Manages RayCluster lifecycle
- **Dynamic GPU Pool**: Automatic GPU allocation per node (no per-node configuration needed)
- **VRAM-Based Scheduling**: Tasks request VRAM, not GPU counts - Ray handles GPU assignment
- **vLLM Inference**: Deploy models via Ray Serve with automatic replica placement
- **REST API**: Submit jobs remotely via REST API (no CLI required)
- **Heterogeneous GPU Support**: Treats all GPUs on a node as a VRAM pool

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

### Dynamic GPU Pool
- Request high GPU count (`nvidia.com/gpu: "10"`) - Kubernetes allocates all available GPUs per node
- No need to specify exact GPU counts per node
- Each GPU node automatically gets all its GPUs

### VRAM-Based Scheduling
- Workers report total VRAM across all GPUs as Ray custom resource
- Tasks request VRAM amount (e.g., `VRAM: 8192` for 8GB model)
- Ray scheduler places tasks based on VRAM availability, not GPU count
- Works seamlessly with heterogeneous GPU configurations

### vLLM Inference
- Deploy models via Ray Serve
- Automatic replica placement based on VRAM
- Treats all GPUs on a node as a single VRAM pool
- Supports tensor parallelism for large models

## Repository Structure

```
ray-k3s-deployment/
├── helm/
│   ├── kuberay-operator-values.yaml    # KubeRay operator Helm values
│   └── install-operator.sh              # Operator installation script
├── manifests/
│   ├── nvidia-device-plugin.yaml       # NVIDIA device plugin (if needed)
│   ├── raycluster.yaml                 # RayCluster CRD with GPU workers
│   └── ray-loadbalancer.yaml           # LoadBalancer service
├── scripts/
│   ├── verify-gpu-resources.sh         # Verify GPU resources are exposed
│   └── ray-vram-setup.sh               # Ray worker VRAM detection script
├── examples/
│   ├── submit-job-rest.py              # REST API job submission example
│   ├── submit-job-curl.sh              # curl-based job submission
│   ├── deploy-vllm-model.py            # vLLM deployment example
│   └── vllm-inference-client.py         # vLLM inference client
├── monitoring/
│   ├── prometheus-scrape-config.yaml   # Prometheus scrape config
│   └── grafana-dashboard.json          # Grafana dashboard (optional)
├── docs/
│   ├── deployment-steps.md             # Step-by-step deployment guide
│   ├── rest-api-submission.md          # REST API documentation
│   └── vllm-deployment.md              # vLLM deployment guide
├── resource-limits.md                  # Hardware specs and recommendations
└── vram-inventory.md                   # Per-node VRAM breakdown
```

## Quick Start

### Prerequisites

- k3s cluster running
- kubectl configured with kubeconfig
- Helm 3.x installed
- MetalLB configured (LoadBalancer IP pool)
- NVIDIA drivers installed on GPU nodes
- NVIDIA device plugin deployed (or deploy via this repo)

### Deployment Steps

1. **Deploy NVIDIA Device Plugin** (if not already present)
   ```bash
   kubectl apply -f manifests/nvidia-device-plugin.yaml
   ```

2. **Install KubeRay Operator**
   ```bash
   cd helm
   ./install-operator.sh
   ```

3. **Deploy RayCluster**
   ```bash
   kubectl apply -f manifests/raycluster.yaml
   ```

4. **Create LoadBalancer Service**
   ```bash
   kubectl apply -f manifests/ray-loadbalancer.yaml
   ```

5. **Get LoadBalancer IP**
   ```bash
   kubectl get svc ray-cluster-loadbalancer -n ray-system
   ```

See [docs/deployment-steps.md](docs/deployment-steps.md) for detailed instructions.

## Usage

### Submit Job via REST API

```python
import requests

response = requests.post(
    "http://<LoadBalancer-IP>:8265/api/jobs",
    json={
        "entrypoint": "python my_script.py",
        "runtime_env": {"pip": ["numpy"]}
    }
)
job_id = response.json()["job_id"]
```

See [docs/rest-api-submission.md](docs/rest-api-submission.md) for complete API reference.

### Deploy vLLM Model

```python
from ray import serve
from vllm import LLM

@serve.deployment(
    ray_actor_options={
        "resources": {"VRAM": 8192}  # Request 8GB VRAM
    }
)
class vLLMModel:
    def __init__(self, model_name: str):
        self.llm = LLM(model=model_name)
    
    def generate(self, prompt: str):
        return self.llm.generate(prompt)

serve.run(vLLMModel.bind("microsoft/phi-2"), name="phi2")
```

See [docs/vllm-deployment.md](docs/vllm-deployment.md) for details.

## How It Works

### Dynamic GPU Allocation

- RayCluster manifest requests high GPU count (`nvidia.com/gpu: "10"`)
- Kubernetes allocates all available GPUs on each node (up to requested amount)
- Each GPU node automatically gets all its GPUs:
  - Node with 3 GPUs → pod gets 3 GPUs
  - Node with 2 GPUs → pod gets 2 GPUs
  - Node with 1 GPU → pod gets 1 GPU

### VRAM Custom Resource

- Worker startup script queries `nvidia-smi` to sum VRAM across all allocated GPUs
- Reports total VRAM as Ray custom resource: `VRAM: <total_mb>`
- Ray scheduler uses VRAM resource for task placement
- Tasks request VRAM amount, Ray handles GPU assignment automatically

### Benefits

- **No per-node configuration**: Same manifest works for all GPU nodes
- **Automatic GPU allocation**: Kubernetes gives all available GPUs per node
- **VRAM-based scheduling**: Tasks request VRAM, not GPU counts
- **Flexible placement**: Ray can use any GPU(s) in the worker's pool
- **Heterogeneous GPU support**: Works with mixed GPU types per node

## Configuration

### GPU Nodes

The deployment automatically handles heterogeneous GPU configurations:
- Each GPU node can have different GPUs (e.g., RTX 3060 Ti, RTX 3090, etc.)
- Workers sum VRAM across all GPUs on the node
- Ray treats all GPUs as a VRAM pool

### Resource Limits

For maximum compute, use resource requests only (no limits):
```yaml
resources:
  requests:
    cpu: "4"
    memory: "8Gi"
    nvidia.com/gpu: "10"  # High number, K8s gives all available
  # No limits - jobs can use all available resources
```

## Monitoring

- **Ray Dashboard**: `http://<LoadBalancer-IP>:8265`
- **Prometheus Metrics**: `/metrics` endpoint
- **Grafana Dashboard**: See `monitoring/grafana-dashboard.json`

## Documentation

- [Deployment Steps](docs/deployment-steps.md) - Complete step-by-step guide
- [REST API Reference](docs/rest-api-submission.md) - Job submission API
- [vLLM Deployment](docs/vllm-deployment.md) - Model serving guide
- [Resource Limits](resource-limits.md) - Hardware specifications
- [VRAM Inventory](vram-inventory.md) - Per-node VRAM breakdown

## Troubleshooting

### GPUs Not Visible
```bash
# Check device plugin
kubectl get pods -n kube-system | grep nvidia-device-plugin

# Verify GPU resources
kubectl describe node <gpu-node> | grep nvidia.com/gpu
```

### Ray Workers Not Starting
```bash
# Check pod events
kubectl describe pod <worker-pod> -n ray-system

# Check logs
kubectl logs <worker-pod> -n ray-system
```

### LoadBalancer IP Not Assigned
```bash
# Check MetalLB
kubectl get pods -n metallb-system

# Check service
kubectl describe svc ray-cluster-loadbalancer -n ray-system
```

## Dependencies

- **KubeRay Operator**: v1.x
- **Ray**: 2.21.0+
- **NVIDIA Device Plugin**: v0.14.1+
- **vLLM**: Latest
- **Helm**: v3.x
- **MetalLB**: Already configured in cluster

## License

[Your License Here]

## Related Repositories

- [rayify](https://github.com/yourusername/rayify) - Ray script conversion tool

