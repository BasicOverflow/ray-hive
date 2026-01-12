"""Deploy models using various strategies."""
import ray
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vram_scheduler import VRAMScheduler

scheduler = VRAMScheduler(suppress_logging=False)

# Deploy with custom vLLM kwargs
# Replicas: one per GPU (capped at number of available GPUs)
# test_mode=True: deploy only on GPU with most VRAM (useful for testing)
# max_num_seqs calculated automatically if not provided
# Model architecture params can be auto-detected from HuggingFace config
# scheduler.deploy_model(
#     model_id="qwen-custom",
#     replicas=1,  # Number of replicas (ignored when test_mode=True)
#     test_mode=True,  # Set to True to deploy only on GPU with most VRAM
#     model_name="Qwen/Qwen3-0.6B-GPTQ-Int8",
#     vram_weights_gb=0.763,  # Model weights only (KV cache calculated separately)
#     max_num_seqs=None,  # Max concurrent sequences per instance (optional, calculated if not provided)
#     max_model_len=2048,  # max prompt length
#     # Architecture params (auto-detected if not provided):
#     # hidden_dim=768,  # Model hidden dimension
#     # num_layers=12,   # Number of transformer layers
#     # dtype="int8",    # Model dtype (int8, fp8, fp16, bf16, fp32)
#     enforce_eager=True,  # pre-allocate necessary buffers to avoid lazy page faults
#     disable_custom_all_reduce=True,  # useful if running single-GPU, disables distributed reduce
#     kv_cache_dtype="fp8",  # reduce KV memory footprint
# )



scheduler.deploy_model(
    model_id="qwen-custom",
    replicas=6,  # Number of replicas (ignored when test_mode=True)
    test_mode=False,  # Set to True to deploy only on GPU with most VRAM
    model_name="Qwen/Qwen3-0.6B-GPTQ-Int8",
    vram_weights_gb=0.763,  # Model weights only (KV cache calculated separately)
    max_num_seqs=None,  # Max concurrent sequences per instance (optional, calculated if not provided)
    max_model_len=2048,  # max prompt length
    # Architecture params (auto-detected if not provided):
    # hidden_dim=768,  # Model hidden dimension
    # num_layers=12,   # Number of transformer layers
    # dtype="int8",    # Model dtype (int8, fp8, fp16, bf16, fp32)
    enforce_eager=True,  # pre-allocate necessary buffers to avoid lazy page faults
    disable_custom_all_reduce=True,  # useful if running single-GPU, disables distributed reduce
    kv_cache_dtype="fp8",  # reduce KV memory footprint
)




ray.shutdown()
