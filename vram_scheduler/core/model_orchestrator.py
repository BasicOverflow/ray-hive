"""Model Orchestrator - deploys models via Ray Serve."""
import ray
from ray import serve
from typing import Dict, Optional, List

from .vllm_model_actor import VLLMModel
from .model_router import ModelRouter

@ray.remote(num_cpus=0)
class ModelOrchestrator:
    """Deploys models via Ray Serve based on configuration."""
    
    VRAM_BUFFER_GB = 0.2
    
    def __init__(self):
        """Initialize orchestrator with state storage for deployment configurations."""
        # Store vLLM args for each deployment: {deployment_name: vllm_args_dict}
        self.deployment_vllm_args: Dict[str, Dict] = {}
    
    def apply(self, model_configs: Dict, vllm_kwargs: Optional[Dict[str, Dict]] = None) -> None:
        """Deploy models. One replica per GPU, max_num_seqs calculated from available VRAM."""
        serve.start()
        
        if vllm_kwargs is None:
            vllm_kwargs = {}
        
        for model_id, config in model_configs.items():
            model_vllm_kwargs = vllm_kwargs.get(model_id, {})
            
            print(f"Deploying {model_id}: {config['name']}, {config['vram_weights_gb']}GB weights")
            
            allocator = ray.get_actor("vram_allocator", namespace="system")
            gpu_info_map = ray.get(allocator.get_all_gpus.remote())
            
            vram_weights_gb = config["vram_weights_gb"]
            max_model_len = config.get("max_model_len", 8192)
            max_tokens = config.get("max_tokens", 256)
            user_max_num_seqs = config.get("max_num_seqs")
            hidden_dim = config.get("hidden_dim")
            num_layers = config.get("num_layers")
            dtype = config.get("dtype")
            max_num_batched_tokens = config.get("max_num_batched_tokens")
            
            available_gpus = []
            for gpu_key, gpu_info in gpu_info_map.items():
                if len(gpu_key) > 50 or gpu_key.startswith('c'):
                    continue
                
                available_gb = gpu_info["available"]
                total_gb = gpu_info["total"]
                available_with_buffer = max(0, available_gb - self.VRAM_BUFFER_GB)
                
                if available_with_buffer >= vram_weights_gb:
                    node_name, gpu_id_str = gpu_key.split(":")
                    gpu_id = gpu_id_str.replace("gpu", "")
                    
                    available_gpus.append({
                        "gpu_key": gpu_key,
                        "resource_name": f"{node_name}_gpu{gpu_id}",
                        "gpu_id": gpu_id,
                        "available_gb": available_gb,
                        "total_gb": total_gb,
                        "available_with_buffer": available_with_buffer
                    })
                    print(f"  GPU {gpu_key}: available {available_gb:.2f}GB (total: {total_gb:.2f}GB)")
            
            if not available_gpus:
                raise RuntimeError(f"No GPUs with sufficient VRAM for {model_id} (need {vram_weights_gb:.2f}GB)")
            
            test_mode = config.get("test_mode", False)
            
            if test_mode:
                best_gpu = max(available_gpus, key=lambda gpu: gpu["available_gb"])
                available_gpus = [best_gpu]
                target_replicas = 1
                print(f"  🧪 TEST MODE: Deploying 1 replica on GPU with most VRAM ({best_gpu['gpu_key']}: {best_gpu['available_gb']:.2f}GB available)")
            else:
                requested_replicas = config.get("replicas")
                target_replicas = min(requested_replicas, len(available_gpus)) if requested_replicas else len(available_gpus)
                if requested_replicas and requested_replicas > len(available_gpus):
                    print(f"  Warning: Requested {requested_replicas} replicas but only {len(available_gpus)} GPUs available. Capping at {len(available_gpus)} replicas.")
                print(f"  Deploying {target_replicas} replicas (one per GPU, max {len(available_gpus)} GPUs available)")
            
            app_name = model_id
            
            # Build all deployments first, then deploy them all at once as a single application
            deployments_dict = {}
            gpu_mapping = {}
            for gpu_info in available_gpus[:target_replicas]:
                gpu_deployment_name = f"{model_id}-{gpu_info['gpu_key'].replace(':', '-').replace('_', '-')}"
                gpu_fraction = max(0.01, round(vram_weights_gb / gpu_info['total_gb'], 2))
                
                # Build full vLLM args dict for this deployment
                deployment_vllm_args = {
                    "model_id": model_id,
                    "model_name": config["name"],
                    "required_vram_weights_gb": vram_weights_gb,
                    "target_gpu_id": gpu_info["gpu_id"],
                    "max_model_len": max_model_len,
                    "max_tokens": max_tokens,
                    "max_num_seqs": user_max_num_seqs,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dtype": dtype,
                    "max_num_batched_tokens": max_num_batched_tokens,
                    **model_vllm_kwargs  # User-provided vLLM kwargs
                }
                
                # Store vLLM args in state
                self.deployment_vllm_args[gpu_deployment_name] = deployment_vllm_args.copy()
                
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
                print(f"    Prepared {gpu_deployment_name} for GPU {gpu_info['gpu_key']}")
            
            # Deploy each GPU as a separate application, then create a router in front
            # Each deployment targets a specific GPU and can have different vLLM args
            # We need separate deployments (not replicas) because each needs different GPU assignment
            if deployments_dict:
                gpu_deployment_names = list(deployments_dict.keys())
                
                # Deploy each GPU deployment as a separate application
                for gpu_deployment_name, deployment in deployments_dict.items():
                    gpu_app_name = f"{app_name}-{gpu_deployment_name}"
                    serve.run(deployment, name=gpu_app_name, route_prefix=f"/{gpu_deployment_name}")
                    print(f"    ✅ Deployed {gpu_deployment_name} on GPU {gpu_mapping[gpu_deployment_name]}")
                
                # Deploy router as the main application that load balances across all GPUs
                router_deployment = ModelRouter.options(
                    name=f"{app_name}-router",
                    autoscaling_config=None,
                    num_replicas=1
                ).bind(
                    model_id=model_id,
                    gpu_deployment_names=gpu_deployment_names
                )
                
                serve.run(router_deployment, name=app_name, route_prefix=f"/{app_name}")
                print(f"    ✅ Deployed router for {app_name} (routing to {len(gpu_deployment_names)} GPU deployments)")
        
        print("All models deployed!")
