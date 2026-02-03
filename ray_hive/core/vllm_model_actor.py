# vLLM Model Actor - one instance per GPU
# Each deployment targets a specific GPU via CUDA_VISIBLE_DEVICES
# Calculates optimal VRAM usage based on vLLM's PagedAttention memory model
# Supports CPU offloading for higher concurrency

import ray
from ray import serve
from typing import Optional, Type, Union, List, Dict
from pydantic import BaseModel
from dataclasses import dataclass, asdict
import json

@dataclass
class DeploymentCalculation:
    """Stores deployment details for routing and display."""
    # GPU specs
    gpu_model: str
    sm_count: Optional[int]
    l2_cache_kb: Optional[float]
    compute_cap: Optional[str]
    total_vram_gb: float
    free_vram_gb: float
    
    # Model info
    model_name: str
    
    # Key configuration
    max_num_seqs: int
    max_num_batched_tokens: int
    
    # CPU offloading (optional)
    swap_space_gb: Optional[float]
    active_seqs_gpu: Optional[int]
    parked_seqs_cpu: Optional[int]
    total_concurrent_seqs: Optional[int]

# Configurable GPU utilization target (default 0.96)
GPU_UTILIZATION_TARGET = 0.96

@serve.deployment(
    ray_actor_options={
        "num_gpus": 0.01,  # Fractional GPU for scheduling
        "memory": 2 * 1024 * 1024 * 1024,  # Request 2GB system RAM
    },
    autoscaling_config={
        "min_replicas": 0,
        "max_replicas": 100,
        "target_num_ongoing_requests_per_replica": 1
    }
)
class VLLMModel:
    """vLLM model - one instance per GPU."""
    
    def __init__(
        self,
        model_id: str,
        model_name: str,
        required_vram_weights_gb: float,
        max_input_prompt_length: int,
        max_output_prompt_length: int,
        target_gpu_id: str = None,
        max_num_seqs: int = None,
        max_num_batched_tokens: int = None,
        gpu_utilization_target: float = 0.96,
        swap_space: float = 0.0,
        **vllm_kwargs
    ):
        self.model_id = model_id
        self.required_vram_weights_gb = required_vram_weights_gb
        self.max_output_prompt_length = max_output_prompt_length
        self.max_tokens = max_output_prompt_length
        self.max_token_response_length = max_output_prompt_length
        self.max_input_prompt_length = max_input_prompt_length
        self.max_prompt_length = max_input_prompt_length
        
        self.max_sequence_length = self.max_prompt_length + self.max_token_response_length
        self.max_model_len = self.max_sequence_length
        
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.gpu_utilization_target = gpu_utilization_target
        self.swap_space = swap_space
        self.calculation_details: Optional[DeploymentCalculation] = None
        self._active_requests = 0
        
        import sys
        import os
        
        os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu_id
        os.environ["VLLM_DISABLE_COMPACT_KV"] = "1"
        os.environ["VLLM_USE_SMALL_WORKSPACE"] = "1"
        os.environ["VLLM_DISABLE_MARLIN"] = "1"
        
        vllm_paths = [
            "/vllm-install",
            "/vllm-install/lib/python3.12/site-packages"
        ]
        for path in vllm_paths:
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)
        
        allocator = ray.get_actor("vram_allocator", namespace="system")
        
        import os
        k8s_node_name = os.getenv("NODE_NAME")
        gpu_key = f"{k8s_node_name}:gpu{target_gpu_id}"
        self.gpu_key = gpu_key
        self.instance_id = f"{model_id}-{gpu_key}"
        
        estimated_total_vram = required_vram_weights_gb + 2.0
        ray.get(allocator.reserve.remote(self.instance_id, gpu_key, estimated_total_vram))
        
        gpu_info = ray.get(allocator.get_gpu_vram.remote(gpu_key))
        
        total_memory_gb = gpu_info["total"]
        available_memory_gb = gpu_info["available"]
        
        from vllm import LLM
        import vllm
        
        # Auto-detect model architecture specs
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        hidden_dim = getattr(config, 'hidden_size', None) or getattr(config, 'd_model', None)
        num_layers = getattr(config, 'num_hidden_layers', None) or getattr(config, 'num_layers', None) or getattr(config, 'n_layer', None)
        num_heads = getattr(config, 'num_attention_heads', None) or getattr(config, 'num_heads', None) or getattr(config, 'n_head', None)
        
        kv_dtype = vllm_kwargs.get("kv_cache_dtype", "fp16")
        dtype = "int8" if kv_dtype in ["int8", "uint8"] else "fp8" if kv_dtype == "fp8" else "fp16"
        
        # Store detected values
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dtype = dtype
                
        # Get GPU specs
        sm_count = gpu_info.get("sm_count")
        gpu_model = gpu_info.get("gpu_model", "Unknown")
        compute_cap = gpu_info.get("compute_cap")
        l2_cache_kb = gpu_info.get("l2_cache_kb")
        
        # Step 2: Break Down VRAM Usage
        dtype_sizes = {
                "int8": 1,
                "uint8": 1,
                "fp8": 1,
                "fp16": 2,
                "bf16": 2,
                "fp32": 4
            }
        dtype_size_bytes = dtype_sizes.get(dtype.lower(), 2)
        
        # Step 2a: Calculate KV cache per sequence (using max_sequence_length instead of max_model_len)
        kv_per_token_per_layer_bytes = 2 * hidden_dim * dtype_size_bytes
        kv_per_seq_bytes = kv_per_token_per_layer_bytes * num_layers * self.max_sequence_length
        kv_per_seq_gb = kv_per_seq_bytes / (1024**3)
        
        activation_per_token_bytes = hidden_dim * dtype_size_bytes * num_layers
        max_num_batched_tokens = self.max_num_batched_tokens
        
        activation_buffer_bytes = activation_per_token_bytes * max_num_batched_tokens
        activation_buffer_gb = activation_buffer_bytes / (1024**3)
        
        # Use gpu_utilization_target for VRAM budget calculation
        gpu_memory_utilization = vllm_kwargs.get("gpu_memory_utilization", self.gpu_utilization_target)
        
        # Calculate VRAM budget
        vram_available = total_memory_gb * self.gpu_utilization_target
        vram_kv_budget_gb = vram_available - required_vram_weights_gb - activation_buffer_gb
        max_num_seqs_memory = max(1, int(vram_kv_budget_gb / kv_per_seq_gb))
        
        swap_space_gb = float(self.swap_space) if self.swap_space > 0 else 0.0
        cpu_offloading_enabled = swap_space_gb > 0
        
        max_num_seqs = self.max_num_seqs
        
        # Apply VRAM constraints and CPU offloading
        if cpu_offloading_enabled:
            active_seqs_gpu = max_num_seqs_memory
            parked_seqs_cpu = int(swap_space_gb / kv_per_seq_gb)
            total_concurrent_seqs = max_num_seqs
        else:
            active_seqs_gpu = None
            parked_seqs_cpu = None
            total_concurrent_seqs = None
        
        # Calculate total VRAM needed
        kv_cache_total_gb = kv_per_seq_gb * (active_seqs_gpu if cpu_offloading_enabled else max_num_seqs)
        total_needed_gb = required_vram_weights_gb + kv_cache_total_gb + activation_buffer_gb
        
        # Prepare vLLM init kwargs
        vllm_init_kwargs = {
            "model": model_name,
            "max_model_len": self.max_model_len,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "swap_space": swap_space_gb if cpu_offloading_enabled else 0,
            "enable_chunked_prefill": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": True,
        }
        
        vllm_init_kwargs.update({k: v for k, v in vllm_kwargs.items() if k != "quantization"})
        
        # Store calculation details
        self.calculation_details = DeploymentCalculation(
            gpu_model=gpu_model,
            sm_count=sm_count,
            l2_cache_kb=l2_cache_kb,
            compute_cap=compute_cap,
            total_vram_gb=total_memory_gb,
            free_vram_gb=available_memory_gb,
            model_name=model_name,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            swap_space_gb=swap_space_gb if cpu_offloading_enabled else None,
            active_seqs_gpu=active_seqs_gpu,
            parked_seqs_cpu=parked_seqs_cpu,
            total_concurrent_seqs=total_concurrent_seqs,
            )
        
        self.llm = LLM(**vllm_init_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        
        ray.get(allocator.release.remote(self.instance_id, gpu_key))
        ray.get(allocator.reserve.remote(self.instance_id, gpu_key, total_needed_gb))
        ray.get(allocator.mark_initialized.remote(self.instance_id, gpu_key))
    
    def __del__(self):
        if hasattr(self, 'instance_id') and hasattr(self, 'gpu_key'):
            allocator = ray.get_actor("vram_allocator", namespace="system")
            ray.get(allocator.release.remote(self.instance_id, self.gpu_key))
    
    
    def __call__(self, request):
        self._active_requests += 1
        try:
            if isinstance(request, dict):
                prompt = request.get("prompts") or request.get("prompt")
                kwargs = {k: v for k, v in request.items() if k not in ["prompt", "prompts"]}
            else:
                prompt = str(request)
                kwargs = {}
            return self.generate(prompt, **kwargs)
        finally:
            self._active_requests = max(0, self._active_requests - 1)
    
    def generate(self, prompt: Union[str, List[str]], **kwargs):
        from vllm import SamplingParams
        
        if isinstance(prompt, str):
            prompts = [prompt]
        elif isinstance(prompt, list):
            prompts = [str(p) for p in prompt]
        
        processed_prompts = prompts
        
        guided_json = kwargs.pop("guided_json", None)
        sampling_kwargs = {
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": kwargs.pop("temperature", 1.0),
            "top_p": kwargs.pop("top_p", 1.0),
            "top_k": kwargs.pop("top_k", -1),
            "stop": kwargs.pop("stop", None),
            "stop_token_ids": kwargs.pop("stop_token_ids", None),
        }
        
        # Use structured_outputs API (vLLM 0.11+)
        if guided_json is not None:
            from vllm.sampling_params import StructuredOutputsParams
            structured_outputs = StructuredOutputsParams(json=guided_json)
            sampling_params = SamplingParams(**sampling_kwargs, structured_outputs=structured_outputs)
        else:
            sampling_params = SamplingParams(**sampling_kwargs)
        
        outputs = self.llm.generate(processed_prompts, sampling_params)
        
        # Extract results from vLLM output format
        # vLLM returns a list of RequestOutput objects, one per input prompt
        results = []
        for output in outputs:
            # Each RequestOutput has an 'outputs' list containing CompletionOutput objects
            if hasattr(output, 'outputs') and output.outputs:
                # Get the first completion output's text
                results.append(output.outputs[0].text)
            elif hasattr(output, 'text'):
                results.append(output.text)
            else:
                results.append(str(output))
        
        return results
    
    def generate_with_schema(self, prompt: str, pydantic_class: Type[BaseModel], **kwargs):
        json_schema = pydantic_class.model_json_schema()
        result = self.generate(prompt, guided_json=json_schema, **kwargs)
        text = result[0] if isinstance(result, list) else str(result)
        return pydantic_class(**json.loads(text))
    
    def get_max_num_seqs(self) -> int:
        """Get the max_num_seqs value for this model instance."""
        return self.max_num_seqs
    
    def get_calculation_details(self) -> Optional[Dict]:
        """Get calculation details as dictionary."""
        if self.calculation_details is None:
            return None
        return asdict(self.calculation_details)
    
    def get_capacity_info(self) -> Dict:
        """Get capacity information for routing decisions."""
        if self.calculation_details is None:
            return {
                "max_num_seqs": self.max_num_seqs,
                "gpu_model": "Unknown",
                "sm_count": None,
                "total_vram_gb": None,
                "approximate_queue_depth": self._active_requests
            }
        
        return {
            "max_num_seqs": self.max_num_seqs,
            "gpu_model": self.calculation_details.gpu_model,
            "sm_count": self.calculation_details.sm_count,
            "total_vram_gb": self.calculation_details.total_vram_gb,
            "approximate_queue_depth": self._active_requests
        }
    

