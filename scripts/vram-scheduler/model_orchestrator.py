"""Model Orchestrator - declarative model deployment."""
import ray
from ray import serve
from typing import Dict, List
import sys
import os

vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
if vram_scheduler_dir not in sys.path:
    sys.path.insert(0, vram_scheduler_dir)

import vllm_model_actor
VLLMModel = vllm_model_actor.VLLMModel

@ray.remote(num_cpus=0)
class ModelOrchestrator:
    """Deploys models via Ray Serve based on configuration."""
    
    # VRAM buffer - hard buffer to leave free on every GPU
    VRAM_BUFFER_GB = 0.4  # 0.4GB buffer per GPU
    
    def _create_router_deployment(self, model_id: str, deployment_names: List[str], app_names: List[str], replica_counts: List[int]):
        """Create a router deployment that forwards requests to all GPU-specific deployments with weighted load balancing."""
        import random
        
        @serve.deployment(
            ray_actor_options={"num_cpus": 0.1},
            autoscaling_config={"min_replicas": 1, "max_replicas": 1}
        )
        class Router:
            def __init__(self):
                self.deployment_handles = []
                self.deployment_names = []
                self.weights = []
                self.request_counts = []
                self.round_robin_counters = []
                connected_replicas = []
                
                total_replicas = sum(replica_counts)
                
                for dep_name, app_name, replicas in zip(deployment_names, app_names, replica_counts):
                    try:
                        handle = serve.get_deployment_handle(dep_name, app_name=app_name)
                        self.deployment_handles.append(handle)
                        self.deployment_names.append(dep_name)
                        connected_replicas.append(replicas)
                        self.request_counts.append(0)
                        self.round_robin_counters.append(0.0)
                        print(f"    Router: Connected to {dep_name} ({replicas} replicas)")
                    except Exception as e:
                        print(f"    Router: Warning - Could not connect to {dep_name}: {e}")
                
                if not self.deployment_handles:
                    raise RuntimeError("Router: No deployment handles available")
                
                # Calculate weights based on connected deployments only
                connected_total = sum(connected_replicas)
                if connected_total > 0:
                    self.weights = [r / connected_total for r in connected_replicas]
                    for name, weight in zip(self.deployment_names, self.weights):
                        print(f"    Router: {name} weight: {weight*100:.1f}%")
                else:
                    # Fallback to equal weights if no replicas info
                    self.weights = [1.0 / len(self.deployment_handles)] * len(self.deployment_handles)
            
            async def __call__(self, request):
                """Forward request using weighted round-robin to ensure all deployments get requests."""
                # Use weighted round-robin: select deployment with lowest (counter / weight)
                # This ensures all deployments get requests proportional to their weights
                idx = min(range(len(self.deployment_handles)), 
                          key=lambda i: self.round_robin_counters[i] / self.weights[i] if self.weights[i] > 0 else float('inf'))
                handle = self.deployment_handles[idx]
                self.request_counts[idx] += 1
                self.round_robin_counters[idx] += 1
                
                prompt = request.get("prompt", "") if isinstance(request, dict) else str(request)
                kwargs = {k: v for k, v in request.items() if k != "prompt"} if isinstance(request, dict) else {}
                result = await handle.generate.remote(prompt, **kwargs)
                # Convert list result to dict format
                if isinstance(result, list) and len(result) > 0:
                    return {"text": result[0] if isinstance(result[0], str) else str(result[0])}
                return {"text": str(result)}
            
            def get_stats(self):
                """Get request distribution statistics."""
                total = sum(self.request_counts)
                if total == 0:
                    return ["No requests processed yet"]
                stats = []
                for name, count, weight in zip(self.deployment_names, self.request_counts, self.weights):
                    pct = (count / total * 100) if total > 0 else 0
                    expected = weight * 100
                    stats.append(f"{name}: {count} requests ({pct:.1f}% actual, {expected:.1f}% expected)")
                return stats
        
        serve.run(
            Router.bind(),
            name=model_id,  # Unified application name for the router
            route_prefix=f"/{model_id}"  # Unified route prefix
        )
    
    def apply(self, model_configs: Dict):
        """Deploy all models from configuration."""
        for model_id, config in model_configs.items():
            print(f"Deploying {model_id}: {config['name']}, {config['vram_gb']}GB, {config['replicas']} replicas")
            
            # Get VRAM allocator to query GPU state
            allocator = ray.get_actor("vram_allocator", namespace="system")
            gpu_info_map = ray.get(allocator.get_all_gpus.remote())
            
            if not gpu_info_map:
                raise RuntimeError("No GPU info available from VRAM allocator")
            
            # Use VRAM requirement directly (no multipliers - already accounts for model + KV cache + overhead)
            required_per_replica_gb = config["vram_gb"]
            
            # Distribute replicas across GPUs based on available VRAM
            # Group GPUs and calculate replicas per GPU
            gpu_deployments = []
            total_replicas_assigned = 0
            remaining_replicas = config["replicas"]
            
            # Sort GPUs by available VRAM (descending) to fill larger GPUs first
            sorted_gpus = sorted(
                gpu_info_map.items(),
                key=lambda x: x[1].get("available", 0),
                reverse=True
            )
            
            for gpu_key, gpu_info in sorted_gpus:
                # Skip old Ray node IDs
                if len(gpu_key) > 50 or gpu_key.startswith('c'):
                    continue
                
                if remaining_replicas <= 0:
                    break
                
                available_gb = gpu_info.get("available", 0)
                total_gb = gpu_info.get("total", 16.0)
                
                # Apply VRAM buffer (hard buffer per GPU)
                buffer_gb = self.VRAM_BUFFER_GB
                available_with_buffer = max(0, available_gb - buffer_gb)
                
                # Calculate how many replicas can fit on this GPU based on available VRAM (with buffer)
                replicas_for_gpu = min(
                    int(available_with_buffer / required_per_replica_gb),
                    remaining_replicas
                )
                
                if replicas_for_gpu > 0:
                    # Create resource name (no colons allowed)
                    node_name, gpu_id_str = gpu_key.split(":")
                    gpu_id = gpu_id_str.replace("gpu", "")
                    resource_name = f"{node_name}_gpu{gpu_id}"
                    
                    # Calculate GPU fraction for this deployment
                    gpu_fraction = required_per_replica_gb / total_gb
                    gpu_fraction = max(gpu_fraction, 0.01)  # Minimum 0.01 to allow scheduling
                    gpu_fraction = round(gpu_fraction, 2)  # Round to 2 decimal places
                    
                    gpu_deployments.append({
                        "gpu_key": gpu_key,
                        "resource_name": resource_name,
                        "gpu_id": gpu_id,  # Store GPU ID to avoid re-extraction
                        "replicas": replicas_for_gpu,
                        "gpu_fraction": gpu_fraction
                    })
                    
                    total_replicas_assigned += replicas_for_gpu
                    remaining_replicas -= replicas_for_gpu
                    
                    print(f"  GPU {gpu_key}: {replicas_for_gpu} replicas (available: {available_gb:.2f}GB, buffer: {buffer_gb:.2f}GB, after buffer: {available_with_buffer:.2f}GB, resource: {resource_name})")
            
            if total_replicas_assigned < config["replicas"]:
                print(f"  WARNING: Only {total_replicas_assigned}/{config['replicas']} replicas can be deployed (insufficient VRAM)")
            
            # Create separate deployment for each GPU with custom resource requests
            # Store deployment and app names for router
            gpu_deployment_names = []
            gpu_app_names = []
            
            for deployment_info in gpu_deployments:
                total_replicas = deployment_info["replicas"]
                gpu_id = deployment_info["gpu_id"]
                deployment_name = f"{model_id}-{deployment_info['gpu_key'].replace(':', '-').replace('_', '-')}"
                app_name = f"{model_id}-gpu-{deployment_info['gpu_key'].replace(':', '-')}"  # Unique app name per GPU
                
                gpu_deployment_names.append(deployment_name)
                gpu_app_names.append(app_name)
                
                print(f"  Creating deployment {deployment_name} with {total_replicas} replicas")
                print(f"    Resource: {deployment_info['resource_name']} = 1")
                print(f"    GPU ID: {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})")
                print(f"    GPU fraction per replica: {deployment_info['gpu_fraction']:.2f}")
                print(f"    Application: {app_name}")
                
                serve.run(
                    VLLMModel.options(
                        name=deployment_name,
                        ray_actor_options={
                            "num_gpus": deployment_info["gpu_fraction"],
                            "memory": 2 * 1024 * 1024 * 1024,
                            "resources": {
                                deployment_info["resource_name"]: 1
                            }
                        },
                        autoscaling_config={
                            "min_replicas": total_replicas,
                            "max_replicas": total_replicas
                        }
                    ).bind(
                        model_id=model_id,
                        model_name=config["name"],
                        required_vram_gb=config["vram_gb"],
                        target_gpu_id=gpu_id
                    ),
                    name=app_name,  # Unique app name per GPU deployment
                    route_prefix=f"/{deployment_name}"  # Unique route for direct access
                )
            
            # Create router deployment that provides unified endpoint (only if multiple GPUs)
            if len(gpu_deployment_names) > 1:
                print(f"  Creating router deployment for unified endpoint: /{model_id}")
                replica_counts = [info["replicas"] for info in gpu_deployments]
                self._create_router_deployment(model_id, gpu_deployment_names, gpu_app_names, replica_counts)
        
        print("All models deployed!")

