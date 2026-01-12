"""vLLM Model Actor - one instance per GPU."""
import ray
from ray import serve
from typing import Optional, Type, Union, List, Dict
from pydantic import BaseModel
import json
import time

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
        target_gpu_id: str = None,
        max_model_len: int = 8192,
        max_tokens: int = 256,
        max_num_seqs: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        dtype: Optional[str] = None,
        max_num_batched_tokens: Optional[int] = None,
        **vllm_kwargs
    ):
        self.model_id = model_id
        self.required_vram_weights_gb = required_vram_weights_gb
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens
        self.max_num_seqs = max_num_seqs
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dtype = dtype
        self.max_num_batched_tokens = max_num_batched_tokens or vllm_kwargs.get("max_num_batched_tokens", 2048)
        
        # Store all vLLM arguments for this deployment
        self.vllm_args = {
            "model_id": model_id,
            "model_name": model_name,
            "required_vram_weights_gb": required_vram_weights_gb,
            "target_gpu_id": target_gpu_id,
            "max_model_len": max_model_len,
            "max_tokens": max_tokens,
            "max_num_seqs": max_num_seqs,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dtype": dtype,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            **vllm_kwargs  # User-provided vLLM kwargs
        }
        
        import sys
        import os
        
        if target_gpu_id is None:
            raise RuntimeError("target_gpu_id must be provided")
        
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
        
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        
        allocator = ray.get_actor("vram_allocator", namespace="system")
        
        import socket
        hostname = socket.gethostname()
        parts = hostname.split("-")
        if len(parts) >= 4 and parts[:3] == ["raycluster", "gpu", "workers"]:
            worker_idx = next((i for i, p in enumerate(parts) if p == "worker"), None)
            k8s_node_name = "-".join(parts[3:worker_idx]) if worker_idx and worker_idx > 2 else "-".join(parts[3:-1] if parts[-1] != "worker" else parts[3:-2])
        else:
            k8s_node_name = hostname
        
        gpu_key = f"{k8s_node_name}:gpu{target_gpu_id}"
        self.gpu_key = gpu_key
        self.instance_id = f"{model_id}-{gpu_key}"
        
        try:
            estimated_total_vram = required_vram_weights_gb + 2.0
            if not ray.get(allocator.reserve.remote(self.instance_id, gpu_key, estimated_total_vram)):
                gpu_info = ray.get(allocator.get_gpu_vram.remote(gpu_key))
                available_gb = gpu_info.get("available", 0) if gpu_info else 0
                raise RuntimeError(f"VRAM reservation failed on {gpu_key}: need {estimated_total_vram:.2f}GB, have {available_gb:.2f}GB available")
            
            gpu_info = ray.get(allocator.get_gpu_vram.remote(gpu_key))
            if not gpu_info:
                raise RuntimeError(f"Could not get GPU VRAM info for {gpu_key}")
            
            total_memory_gb = gpu_info["total"]
            available_memory_gb = gpu_info["available"]
            
            from vllm import LLM
            
            if hidden_dim is None or num_layers is None:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                if hidden_dim is None:
                    hidden_dim = getattr(config, 'hidden_size', None) or getattr(config, 'd_model', None)
                    if hidden_dim:
                        print(f"[{model_id}] Auto-detected hidden_dim={hidden_dim}", flush=True)
                if num_layers is None:
                    num_layers = getattr(config, 'num_hidden_layers', None) or getattr(config, 'num_layers', None) or getattr(config, 'n_layer', None)
                    if num_layers:
                        print(f"[{model_id}] Auto-detected num_layers={num_layers}", flush=True)
            
            if dtype is None:
                kv_dtype = vllm_kwargs.get("kv_cache_dtype", "fp16")
                dtype = "int8" if kv_dtype in ["int8", "uint8"] else "fp8" if kv_dtype == "fp8" else "fp16"
            
            dtype_sizes = {
                "int8": 1,
                "uint8": 1,
                "fp8": 1,
                "fp16": 2,
                "bf16": 2,
                "fp32": 4
            }
            dtype_size_bytes = dtype_sizes.get(dtype.lower(), 2)  # Default to fp16
            
            kv_per_token_per_layer = 2 * hidden_dim * dtype_size_bytes
            kv_per_token_all_layers = kv_per_token_per_layer * num_layers
            kv_per_seq_bytes = kv_per_token_all_layers * max_model_len
            kv_cache_per_seq_gb = kv_per_seq_bytes / (1024**3)
            print(f"[{model_id}] KV cache calculation: hidden_dim={hidden_dim}, num_layers={num_layers}, dtype_size={dtype_size_bytes}, max_model_len={max_model_len}", flush=True)
            print(f"[{model_id}] KV cache per sequence: {kv_cache_per_seq_gb:.6f}GB ({kv_per_seq_bytes / (1024**2):.2f}MB)", flush=True)
            
            activation_per_token_bytes = hidden_dim * dtype_size_bytes * num_layers
            activation_buffer_bytes = activation_per_token_bytes * self.max_num_batched_tokens
            activation_buffer_gb = activation_buffer_bytes / (1024**3)
            
            if max_num_seqs is None:
                overhead_gb = 0.05
                available_for_kv = max(0, available_memory_gb - required_vram_weights_gb - activation_buffer_gb - overhead_gb)
                max_num_seqs = max(1, int(available_for_kv / kv_cache_per_seq_gb))
                print(f"[{model_id}] ✅ Calculated max_num_seqs={max_num_seqs} for GPU {gpu_key} (available={available_memory_gb:.2f}GB, KV/seq={kv_cache_per_seq_gb:.3f}GB)", flush=True)
            else:
                print(f"[{model_id}] Using provided max_num_seqs={max_num_seqs} for GPU {gpu_key}", flush=True)
            
            # Ensure max_num_seqs is set
            self.max_num_seqs = max_num_seqs
            
            vllm_init_kwargs = {
                "model": model_name,
                "max_model_len": max_model_len,
                "enforce_eager": True,
                "max_num_seqs": max_num_seqs,
                "swap_space": 0,
                "enable_chunked_prefill": False,
            }
            
            kv_cache_total_gb = kv_cache_per_seq_gb * max_num_seqs
            overhead_gb = 0.05
            total_needed_gb = required_vram_weights_gb + kv_cache_total_gb + activation_buffer_gb + overhead_gb
            print(f"[{model_id}] VRAM breakdown: weights={required_vram_weights_gb:.3f}GB, KV cache={kv_cache_total_gb:.3f}GB, activation={activation_buffer_gb:.3f}GB, total={total_needed_gb:.3f}GB", flush=True)
            
            # Cap at 98% utilization (leaves ~2% buffer, ~0.5GB on 24GB GPU)
            calculated_utilization = min(0.98, round(total_needed_gb / total_memory_gb, 3))
            
            if "gpu_memory_utilization" not in vllm_kwargs:
                vllm_init_kwargs["gpu_memory_utilization"] = calculated_utilization
                print(f"[{model_id}] Calculated gpu_memory_utilization={calculated_utilization:.3f} (weights={required_vram_weights_gb:.3f}GB, KV cache={kv_cache_total_gb:.3f}GB, activation={activation_buffer_gb:.3f}GB, total={total_needed_gb:.3f}GB)", flush=True)
            
            invalid_params = {"quantization"}
            for key, value in vllm_kwargs.items():
                if key not in invalid_params:
                    vllm_init_kwargs[key] = value
            
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    self.llm = LLM(**vllm_init_kwargs)
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    if ("rate limit" in error_str or "429" in error_str) and attempt < max_retries - 1:
                        import random
                        wait_time = (2 ** attempt) * 10 + random.uniform(0, 5)
                        print(f"[{model_id}] Rate limit, retrying in {wait_time:.1f}s...", flush=True)
                        time.sleep(wait_time)
                        continue
                    if attempt == max_retries - 1:
                        print(f"[{model_id}] ❌ vLLM initialization failed with kwargs: {list(vllm_init_kwargs.keys())}", flush=True)
                        print(f"[{model_id}] Error: {e}", flush=True)
                    raise
            
            try:
                self.tokenizer = self.llm.get_tokenizer()
            except:
                self.tokenizer = None
            
            ray.get(allocator.release.remote(self.instance_id, gpu_key))
            ray.get(allocator.reserve.remote(self.instance_id, gpu_key, total_needed_gb))
            ray.get(allocator.mark_initialized.remote(self.instance_id, gpu_key))
            print(f"[{model_id}] ✅ Model loaded on {gpu_key} with max_num_seqs={max_num_seqs}, total VRAM={total_needed_gb:.3f}GB", flush=True)
            
        except Exception as e:
            ray.get(allocator.release.remote(self.instance_id, self.gpu_key))
            raise
    
    def __del__(self):
        """Cleanup: release VRAM reservation."""
        if hasattr(self, 'instance_id') and hasattr(self, 'gpu_key'):
            try:
                allocator = ray.get_actor("vram_allocator", namespace="system")
                ray.get(allocator.release.remote(self.instance_id, self.gpu_key))
            except:
                pass
    
    def _truncate_input(self, prompt: str) -> str:
        """Truncate prompt to fit within max_model_len."""
        if self.tokenizer is None:
            return prompt
        
        buffer = 50
        max_input_tokens = self.max_model_len - self.max_tokens - buffer
        
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) <= max_input_tokens:
            return prompt
        
        truncated_tokens = tokens[:max_input_tokens]
        return self.tokenizer.decode(truncated_tokens)
    
    def __call__(self, request):
        """Ray Serve entry point."""
        if isinstance(request, dict):
            # Handle batch mode (prompts) or single mode (prompt)
            if "prompts" in request:
                prompt = request["prompts"]  # Batch mode - list of prompts
            elif "prompt" in request:
                prompt = request["prompt"]  # Single prompt
            else:
                raise ValueError("Request must contain either 'prompt' or 'prompts' key")
            
            # Extract kwargs (exclude prompt/prompts)
            kwargs = {k: v for k, v in request.items() if k not in ["prompt", "prompts"]}
        else:
            # Non-dict request - treat as single prompt string
            prompt = str(request)
            kwargs = {}
        
        return self.generate(prompt, **kwargs)
    
    def generate(self, prompt: Union[str, List[str]], **kwargs):
        """Generate text using the model. Accepts single prompt or list of prompts."""
        from vllm import SamplingParams
        
        # Handle both single prompt and batch
        if isinstance(prompt, str):
            prompts = [prompt]
        elif isinstance(prompt, list):
            # Ensure it's a proper Python list and convert all to strings
            prompts = [str(p) for p in list(prompt)]
        else:
            prompts = [str(prompt)]
        
        # Validate we have prompts
        if not prompts:
            raise ValueError("Cannot generate with empty prompt list")
        
        # Truncate inputs if needed - preserve all prompts
        processed_prompts = []
        for p in prompts:
            truncated = self._truncate_input(p)
            processed_prompts.append(truncated)
        
        # Extract sampling parameters from kwargs
        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            stop=kwargs.get("stop", None),
            stop_token_ids=kwargs.get("stop_token_ids", None),
        )
        
        # Call vLLM generate with processed prompts
        # vLLM's generate expects: generate(prompts: List[str], sampling_params: SamplingParams)
        # Ensure prompts is a proper list of strings (not a tuple or other iterable)
        if not isinstance(processed_prompts, list):
            processed_prompts = list(processed_prompts)
        
        # Call vLLM generate - it handles batching internally
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
        """Generate structured output using Pydantic schema."""
        schema_str = json.dumps(pydantic_class.model_json_schema(), indent=2)
        enhanced_prompt = f"{prompt}\n\nRespond in valid JSON matching this schema:\n{schema_str}"
        
        result = self.generate(enhanced_prompt, **kwargs)
        text = result[0] if isinstance(result, list) and result else str(result)
        
        import re
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                return pydantic_class(**parsed)
            except:
                pass
        
        try:
            parsed = json.loads(text)
            return pydantic_class(**parsed)
        except:
            raise ValueError(f"Could not parse structured output from: {text}")
    
    def get_max_num_seqs(self) -> int:
        """Get the max_num_seqs value for this model instance."""
        return self.max_num_seqs
    
    def generate_streaming(self, prompt: str, **kwargs):
        """Generate streaming output (async generator)."""
        raise NotImplementedError("Streaming not yet implemented")

