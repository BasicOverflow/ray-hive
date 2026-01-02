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

### VRAM-Aware Scheduling
- **Dynamic VRAM tracking**: DaemonSet monitors VRAM on each GPU node every 0.5s
- **Global allocator actor**: Singleton actor maintains VRAM state across all nodes
- **Exact VRAM requirements**: Models declare exact VRAM needs, no overcommit
- **Automatic placement**: Ray Serve places replicas based on available VRAM

### vLLM Model Deployment
- Deploy models via Ray Serve with VRAM reservation
- **Multiple models per GPU**: Fractional GPU allocation enables multiple replicas to share a single GPU
- Declarative model configuration
- Automatic scaling and placement
- Zero OOM guarantees through hard reservations

### Cluster Testing
- Basic connectivity and resource tests
- VRAM allocator verification
- CPU/GPU stress testing

## Repository Structure

```
ray-k3s-deployment/
├── manifests/                          # Kubernetes manifests
│   ├── raycluster.yaml                 # Main RayCluster deployment
│   ├── ray-vram-monitor-daemonset.yaml # VRAM monitoring DaemonSet
│   ├── vram-scheduler-configmap.yaml   # VRAM scheduler scripts
│   └── helm/                           # KubeRay operator Helm config
├── scripts/
│   ├── vram-scheduler/                 # VRAM-aware scheduling system
│   │   ├── 1_deploy_models.py         # Main: Deploy models from config
│   │   ├── 2_deploy_max_llms.py       # Alternative: Deploy max replicas
│   │   ├── 3_shutdown_models.py       # Shutdown deployments
│   │   ├── 4_test_inference.py        # Test inference on models
│   │   ├── vram_allocator.py          # Global VRAM allocator actor
│   │   ├── vllm_model_actor.py        # vLLM model with VRAM reservation
│   │   └── model_orchestrator.py      # Declarative model deployment
│   │   # Note: vram_monitor.py is in ConfigMap (see below)
│   └── stress_tests/                   # Cluster testing scripts
│       ├── test_basic_connection.py   # Basic connectivity test
│       └── test_vram_resource.py      # VRAM allocator test
```

## Quick Start

### Deploy Ray Cluster

```bash
# Deploy KubeRay operator (if not already installed)
kubectl apply -f manifests/helm/kuberay-operator-values.yaml

# Deploy Ray cluster
kubectl apply -f manifests/raycluster.yaml

# Deploy VRAM monitor
kubectl apply -f manifests/vram-scheduler-configmap.yaml
kubectl apply -f manifests/ray-vram-monitor-daemonset.yaml
```

### Deploy Models

**Option 1: Deploy from config (recommended)**
```bash
python scripts/vram-scheduler/1_deploy_models.py
```

**Option 2: Deploy maximum replicas**
```bash
python scripts/vram-scheduler/2_deploy_max_llms.py
# Or with custom model:
MODEL_NAME="microsoft/phi-2" MODEL_VRAM_GB=3.0 python scripts/vram-scheduler/2_deploy_max_llms.py
```

### Test Cluster

```bash
# Basic connectivity
python scripts/stress_tests/test_basic_connection.py

# VRAM allocator
python scripts/stress_tests/test_vram_resource.py
```

## How It Works

### VRAM Scheduler Architecture

1. **DaemonSet** runs on each GPU node, queries VRAM via `nvidia-smi` every 0.5s
2. **Global allocator actor** maintains VRAM state across all nodes (named + detached)
3. **Model actors** reserve VRAM before loading, preventing OOM
4. **Ray Serve** places replicas based on VRAM availability

### Multiple Models Per GPU

The system enables multiple vLLM replicas to share a single GPU through fractional GPU allocation (`num_gpus: 0.01`) and CUDA memory slicing. Each replica uses `torch.cuda.set_per_process_memory_fraction()` to hard-limit its VRAM usage before loading, with vLLM configured via `gpu_memory_utilization`. The Ray cluster manifest includes an initContainer to install vllm into a shared volume, CUDA environment variables (`PYTORCH_ALLOC_CONF`, `CUDA_DEVICE_MAX_CONNECTIONS`, `NCCL_P2P_DISABLE`, `NCCL_IB_DISABLE`) to prevent greedy memory allocation, privileged security context for GPU access, and host device mounts for NVIDIA drivers.

### Components

- **`vram_allocator.py`**: Global VRAM state actor (singleton, HA-safe)
- **`vram_monitor.py`**: DaemonSet script (stored in ConfigMap, see below)
- **`vllm_model_actor.py`**: vLLM deployment with VRAM reservation
- **`model_orchestrator.py`**: Declarative model deployment manager

### VRAM Monitor Script (ConfigMap)

The `vram_monitor.py` script runs in the DaemonSet and is stored in the ConfigMap. Here's the script:

```python
"""
VRAM Monitor Script - runs in DaemonSet to update allocator.

This script runs on each GPU node and continuously updates
the global VRAM allocator with current VRAM state.
"""
import ray
import subprocess
import time
import sys
import os

# Add scripts directory to path for imports
sys.path.insert(0, "/scripts/vram-scheduler")

from vram_allocator import get_vram_allocator

def get_vram_gb():
    """Get VRAM using nvidia-smi (mounted from host)."""
    try:
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = '/host/usr/lib/x86_64-linux-gnu:' + env.get('LD_LIBRARY_PATH', '')
        # Get free VRAM
        result_free = subprocess.run(
            ['/host/usr/bin/nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=2, env=env
        )
        # Get total VRAM
        result_total = subprocess.run(
            ['/host/usr/bin/nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=2, env=env
        )
        free_values_mb = [int(x.strip()) for x in result_free.stdout.strip().split('\n') if x.strip()]
        total_values_mb = [int(x.strip()) for x in result_total.stdout.strip().split('\n') if x.strip()]
        free_gb = sum(free_values_mb) / 1024.0
        total_gb = sum(total_values_mb) / 1024.0
        return free_gb, total_gb
    except Exception as e:
        print(f"Error getting VRAM: {e}", file=sys.stderr, flush=True)
        return None, None

def main():
    # Connect to Ray cluster
    ray_address = os.getenv("RAY_ADDRESS", "ray://10.0.1.53:10001")
    ray.init(address=ray_address, ignore_reinit_error=True)
    
    # Get the global allocator
    allocator = get_vram_allocator()
    
    print("VRAM Monitor started - updating allocator every 0.5 seconds", 
          file=sys.stderr, flush=True)
    
    while True:
        try:
            # Get actual VRAM
            free_gb, total_gb = get_vram_gb()
            
            if free_gb is not None and total_gb is not None:
                # Use Kubernetes node name as identifier
                k8s_node_name = os.getenv("NODE_NAME", "unknown")
                
                # Update allocator with K8s node name as the key
                ray.get(allocator.update_node.remote(k8s_node_name, free_gb, total_gb))
                
                print(f"K8s Node {k8s_node_name}: {free_gb:.2f}GB free / {total_gb:.2f}GB total", 
                      file=sys.stderr, flush=True)
            else:
                print("Warning: Could not get VRAM", file=sys.stderr, flush=True)
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            time.sleep(1)

if __name__ == "__main__":
    main()
```

## Configuration

- **Ray Address**: `ray://10.0.1.53:10001` (LoadBalancer IP)
- **Dashboard**: `http://10.0.1.53:8265`
- **Cluster**: 6 worker nodes (3 CPU + 3 GPU), 6 GPUs total
- **VRAM Tracking**: Per-node tracking via K8s node names

## Troubleshooting

When deploying multiple replicas that share a GPU, transient memory errors during initialization are expected. Ray Serve automatically retries failed deployments until models successfully load. If models consistently fail, verify VRAM requirements include a 70% buffer for overhead and that total VRAM doesn't exceed available GPU memory.

## Related Repositories

- [rayify](https://github.com/BasicOverflow/rayify) - Ray script conversion tool

