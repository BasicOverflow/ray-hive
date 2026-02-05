import ray
from ray import serve
from typing import Dict, Optional

@ray.remote(num_cpus=0)
class ModelOrchestrator:
    def apply(self, model_configs: Dict, vllm_kwargs: Optional[Dict[str, Dict]] = None) -> Dict[str, Dict]:
        from .vllm_model_actor import VLLMModel
        from .model_router import ModelRouter
        
        serve.start()
        
        if vllm_kwargs is None:
            vllm_kwargs = {}
        
        results = {}
        for model_id, config in model_configs.items():
            model_vllm_kwargs = vllm_kwargs.get(model_id, {})
            
            allocator = ray.get_actor("vram_allocator", namespace="system")
            gpu_info_map = ray.get(allocator.get_all_gpus.remote())
            
            vram_weights_gb = config["vram_weights_gb"]
            max_input_prompt_length = config["max_input_prompt_length"]
            max_output_prompt_length = config["max_output_prompt_length"]
            max_num_seqs = config["max_num_seqs"]
            max_num_batched_tokens = config["max_num_batched_tokens"]
            gpu_utilization_target = config.get("gpu_utilization_target", 0.96)
            swap_space_per_instance = config.get("swap_space_per_instance", 0)
            
            available_gpus = []
            for gpu_key, gpu_info in gpu_info_map.items():
                if len(gpu_key) > 50 or gpu_key.startswith('c'):
                    continue
                
                if gpu_info["available"] >= vram_weights_gb:
                    node_name, gpu_id_str = gpu_key.split(":")
                    gpu_id = gpu_id_str.replace("gpu", "")
                    available_gpus.append({
                        "gpu_key": gpu_key,
                        "resource_name": f"{node_name}_gpu{gpu_id}",
                        "gpu_id": gpu_id,
                        "total_gb": gpu_info["total"]
                    })
            
            gpu = config.get("gpu")
            if gpu is not None:
                if isinstance(gpu, str):
                    available_gpus = [g for g in available_gpus if g["gpu_key"] == gpu]
                    if not available_gpus:
                        raise ValueError(f"GPU {gpu} not found or does not have sufficient VRAM")
                    target_replicas = 1
                elif isinstance(gpu, list):
                    requested_replicas = config.get("replicas")
                    if isinstance(requested_replicas, int):
                        if len(gpu) != requested_replicas:
                            raise ValueError(f"Length of gpu list ({len(gpu)}) must match num_replicas ({requested_replicas})")
                    available_gpus = [g for g in available_gpus if g["gpu_key"] in gpu]
                    if len(available_gpus) != len(gpu):
                        missing = set(gpu) - {g["gpu_key"] for g in available_gpus}
                        raise ValueError(f"Some requested GPUs not found or do not have sufficient VRAM: {missing}")
                    target_replicas = len(gpu)
                else:
                    raise ValueError("gpu must be a string, list of strings, or None")
            else:
                requested_replicas = config.get("replicas")
                if isinstance(requested_replicas, str) and requested_replicas.lower() == 'max':
                    target_replicas = len(available_gpus)
                elif requested_replicas is None:
                    target_replicas = len(available_gpus)
                else:
                    target_replicas = min(requested_replicas, len(available_gpus))
                        
            swap_space_per_gpu = float(swap_space_per_instance) if swap_space_per_instance else 0.0
            
            deployments_dict = {}
            gpu_mapping = {}
            resource_mapping = {}
            for gpu_info in available_gpus[:target_replicas]:
                gpu_deployment_name = f"{model_id}-{gpu_info['gpu_key'].replace(':', '-').replace('_', '-')}"
                gpu_fraction = max(0.01, round(vram_weights_gb / gpu_info['total_gb'], 2))
                
                # Auto-calculate max_num_seqs and max_num_batched_tokens if set to "auto"
                gpu_max_num_seqs = max_num_seqs
                gpu_max_num_batched_tokens = max_num_batched_tokens
                
                if max_num_seqs == "auto":
                    gpu_max_num_seqs = int(850 * (gpu_info['total_gb'] / 24.0))
                
                if max_num_batched_tokens == "auto":
                    gpu_max_num_batched_tokens = int(16384 * (gpu_info['total_gb'] / 24.0))
                
                deployment_vllm_args = {
                    "model_id": model_id,
                    "model_name": config["name"],
                    "required_vram_weights_gb": vram_weights_gb,
                    "target_gpu_id": gpu_info["gpu_id"],
                    "max_input_prompt_length": max_input_prompt_length,
                    "max_output_prompt_length": max_output_prompt_length,
                    "max_num_seqs": gpu_max_num_seqs,
                    "max_num_batched_tokens": gpu_max_num_batched_tokens,
                    "gpu_utilization_target": gpu_utilization_target,
                    "swap_space": swap_space_per_gpu,
                    **model_vllm_kwargs
                }
                
                deployment = VLLMModel.options(
                    name=gpu_deployment_name,
                    ray_actor_options={
                        "num_gpus": gpu_fraction,
                        "memory": 2 * 1024 * 1024 * 1024,
                        "resources": {gpu_info["resource_name"]: 0.01}
                    },
                    autoscaling_config=None,
                    num_replicas=1
                ).bind(**deployment_vllm_args)
                
                deployments_dict[gpu_deployment_name] = deployment
                gpu_mapping[gpu_deployment_name] = gpu_info['gpu_key']
                resource_mapping[gpu_deployment_name] = gpu_info["resource_name"]
            
            if deployments_dict:
                gpu_deployment_names = list(deployments_dict.keys())
                
                # Deploy all GPU deployments in parallel using Ray tasks with node-specific placement
                @ray.remote
                def deploy_single(deployment, deployment_name: str, route_prefix: str):
                    """Deploy a single deployment in a Ray task."""
                    serve.run(deployment, name=deployment_name, route_prefix=route_prefix)
                    return True
                
                # Start all deployments in parallel with node-specific resource placement
                deploy_futures = []
                for gpu_deployment_name, deployment in deployments_dict.items():
                    resource_name = resource_mapping[gpu_deployment_name]
                    future = deploy_single.options(
                        resources={resource_name: 0.01}
                    ).remote(
                        deployment, 
                        gpu_deployment_name, 
                        f"/{gpu_deployment_name}"
                    )
                    deploy_futures.append(future)
                
                # Wait for all deployments to complete in parallel
                ray.get(deploy_futures)
                
                router_deployment = ModelRouter.options(
                    name=f"{model_id}-router",
                    autoscaling_config=None,
                    num_replicas=1
                ).bind(
                    model_id=model_id,
                    gpu_deployment_names=gpu_deployment_names
                )
                
                serve.run(router_deployment, name=model_id, route_prefix=f"/{model_id}")
                
                results[model_id] = {
                    gpu_deployment_name: {
                        "calc": serve.get_deployment_handle(gpu_deployment_name, app_name=gpu_deployment_name).get_calculation_details.remote().result(),
                        "gpu_key": gpu_mapping[gpu_deployment_name]
                    }
                    for gpu_deployment_name in gpu_deployment_names
                }
        
        return results
