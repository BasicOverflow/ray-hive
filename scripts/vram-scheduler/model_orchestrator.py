"""Model Orchestrator - declarative model deployment."""
import ray
from ray import serve
from typing import Dict, List
import sys
import os
import time

vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
if vram_scheduler_dir not in sys.path:
    sys.path.insert(0, vram_scheduler_dir)

import vllm_model_actor
VLLMModel = vllm_model_actor.VLLMModel

@ray.remote(num_cpus=0)
class ModelOrchestrator:
    """Deploys models via Ray Serve based on configuration."""
    
    # VRAM buffer - hard buffer to leave free on every GPU
    VRAM_BUFFER_GB = 0.5  # 0.5GB buffer per GPU
    
    def _create_router_deployment(self, model_id: str, deployment_names: List[str], app_names: List[str]):
        """Create a router deployment that forwards requests to all GPU-specific deployments.
        
        Uses Ray Serve's built-in load balancing: deployment handles automatically load balance
        across replicas within each deployment. The router just selects which deployment handle
        to use, and Ray Serve handles the rest.
        """
        import random
        
        @serve.deployment(
            ray_actor_options={"num_cpus": 0.1},
            autoscaling_config={"min_replicas": 1, "max_replicas": 1}
        )
        class Router:
            def __init__(self):
                self.deployment_handles = []
                # Get handles to all GPU deployments
                # Ray Serve handles automatically load balance across replicas within each deployment
                for dep_name, app_name in zip(deployment_names, app_names):
                    try:
                        handle = serve.get_deployment_handle(dep_name, app_name=app_name)
                        self.deployment_handles.append(handle)
                        print(f"    Router: Connected to {dep_name} in {app_name}")
                    except Exception as e:
                        print(f"    Router: Warning - Could not connect to {dep_name}: {e}")
                
                if not self.deployment_handles:
                    raise RuntimeError("Router: No deployment handles available")
                
                print(f"    Router: Load balancing across {len(self.deployment_handles)} deployments")
            
            async def __call__(self, request):
                """Forward request to a random deployment handle.
                
                Ray Serve's deployment handles automatically load balance across replicas
                within each deployment, so we just need to select which deployment to use.
                """
                if not self.deployment_handles:
                    return {"error": "No deployments available"}
                
                # Random selection across deployments (Ray Serve handles load balancing within each)
                handle = random.choice(self.deployment_handles)
                return await handle.generate.remote(request)
        
        serve.run(
            Router.bind(),
            name=model_id,  # Unified application name for the router
            route_prefix=f"/{model_id}"  # Unified route prefix
        )
    
    def _wait_for_deployment_ready(self, deployment_name: str, expected_replicas: int, gpu_key: str = None, timeout: int = 300) -> bool:
        """Wait for deployment to have expected number of replicas fully initialized (RUNNING).
        
        Also waits for pending VRAM reservations to clear on the GPU to ensure replicas are fully loaded.
        Returns True if deployment is ready, False if timeout.
        """
        start_time = time.time()
        check_interval = 5
        max_checks = timeout // check_interval
        
        allocator = ray.get_actor("vram_allocator", namespace="system")
        
        for check_num in range(max_checks):
            try:
                status = serve.status()
                
                # Check all applications for this deployment
                for app_name, app_info in status.applications.items():
                    if hasattr(app_info, 'deployments') and deployment_name in app_info.deployments:
                        dep = app_info.deployments[deployment_name]
                        
                        # Check if deployment exists and has replicas
                        if hasattr(dep, 'status'):
                            dep_status_str = str(dep.status).upper()
                            if 'ERROR' in dep_status_str or 'FAILED' in dep_status_str:
                                print(f"    Deployment {deployment_name} is in error state")
                                return False
                        
                        # Count RUNNING replicas (fully initialized) and check for errors
                        running = 0
                        error_count = 0
                        if hasattr(dep, 'replicas') and dep.replicas:
                            for replica in dep.replicas:
                                if hasattr(replica, 'state'):
                                    state = str(replica.state).upper()
                                    if state == 'RUNNING':
                                        running += 1
                                    elif 'ERROR' in state or 'FAILED' in state:
                                        error_count += 1
                        
                        if error_count > 0:
                            print(f"    ⚠️  Deployment {deployment_name} has {error_count} replicas in error state")
                            return False
                        
                        # Check pending reservations for status reporting only
                        pending_gb = 0
                        if gpu_key:
                            try:
                                gpu_info = ray.get(allocator.get_gpu_vram.remote(gpu_key))
                                if gpu_info:
                                    pending_gb = gpu_info.get("pending", 0)
                            except:
                                pass
                        
                        if check_num % 3 == 0:  # Print every 3 checks (every 15s)
                            print(f"    Deployment {deployment_name}: {running}/{expected_replicas} RUNNING, {pending_gb:.2f}GB pending on {gpu_key}")
                        
                        # Wait for all replicas to be RUNNING
                        # If replicas are RUNNING, they've already called mark_initialized() and moved from pending to active
                        if running >= expected_replicas:
                            print(f"    ✅ All {expected_replicas} replicas RUNNING on {gpu_key}")
                            return True
                
                time.sleep(check_interval)
            except Exception as e:
                time.sleep(check_interval)
        
        # Final check
        try:
            status = serve.status()
            for app_name, app_info in status.applications.items():
                if hasattr(app_info, 'deployments') and deployment_name in app_info.deployments:
                    dep = app_info.deployments[deployment_name]
                    running = 0
                    if hasattr(dep, 'replicas') and dep.replicas:
                        for replica in dep.replicas:
                            if hasattr(replica, 'state') and str(replica.state).upper() == 'RUNNING':
                                running += 1
                    if running >= expected_replicas:
                        print(f"    Deployment {deployment_name}: {running} replicas RUNNING (proceeding)")
                        return True
        except:
            pass
        
        print(f"    WARNING: Timeout waiting for {deployment_name} to be fully ready")
        return False
    
    def apply(self, model_configs: Dict):
        """Deploy all models from configuration."""
        for model_id, config in model_configs.items():
            print(f"Deploying {model_id}: {config['name']}, {config['vram_gb']}GB, {config['replicas']} replicas")
            
            # Get VRAM allocator to query GPU state
            allocator = ray.get_actor("vram_allocator", namespace="system")
            gpu_info_map = ray.get(allocator.get_all_gpus.remote())
            
            if not gpu_info_map:
                print(f"  WARNING: No GPU info available, falling back to single deployment")
                # Fallback to single deployment
                serve.run(
                    VLLMModel.options(
                        name=model_id,
                        ray_actor_options={
                            "num_gpus": 0.01,
                            "memory": 2 * 1024 * 1024 * 1024,
                        },
                        autoscaling_config={
                            "min_replicas": config["replicas"],
                            "max_replicas": config["replicas"]
                        }
                    ).bind(
                        model_id=model_id,
                        model_name=config["name"],
                        required_vram_gb=config["vram_gb"]
                    ),
                    name=model_id,
                    route_prefix=f"/{model_id}"
                )
                continue
            
            # Use VRAM requirement directly (no multipliers - already accounts for model + KV cache + overhead)
            required_per_replica_gb = config["vram_gb"]
            required_per_replica_mb = int(required_per_replica_gb * 1024)  # Convert to MB for resource request
            
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
                available_mb = int(available_gb * 1024)
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
                    # Use actual VRAM requirement (no conservative caps)
                    total_gb = gpu_info.get("total", 16.0)
                    gpu_fraction = (required_per_replica_gb * replicas_for_gpu) / total_gb / replicas_for_gpu
                    gpu_fraction = max(gpu_fraction, 0.01)  # Minimum 0.01 to allow scheduling
                    gpu_fraction = round(gpu_fraction, 2)  # Round to 2 decimal places
                    
                    gpu_deployments.append({
                        "gpu_key": gpu_key,
                        "resource_name": resource_name,
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
                deployment_name = f"{model_id}-{deployment_info['gpu_key'].replace(':', '-').replace('_', '-')}"
                app_name = f"{model_id}-gpu-{deployment_info['gpu_key'].replace(':', '-')}"  # Unique app name per GPU
                
                # Extract GPU ID from gpu_key (e.g., "ergos-02-nv:gpu2" -> "2")
                gpu_key = deployment_info["gpu_key"]
                _, gpu_id_str = gpu_key.split(":")
                gpu_id = gpu_id_str.replace("gpu", "")
                
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
            
            # Create router deployment that provides unified endpoint
            # Router uses Ray Serve's built-in load balancing via deployment handles
            if len(gpu_deployment_names) > 1:
                print(f"  Creating router deployment for unified endpoint: /{model_id}")
                self._create_router_deployment(model_id, gpu_deployment_names, gpu_app_names)
            elif len(gpu_deployment_names) == 1:
                # Single deployment - no router needed, but create alias for consistency
                print(f"  Single deployment - accessible at /{gpu_deployment_names[0]} or /{model_id}")
                # Optionally create router anyway for consistent API
                self._create_router_deployment(model_id, gpu_deployment_names, gpu_app_names)
        
        print("All models deployed!")

