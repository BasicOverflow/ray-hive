import ray
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ray_hive import RayHive
from ray_hive.inference import inference_batch
from ray import serve

scheduler = RayHive(suppress_logging=True)


# Test configurations - system auto-optimizes based on P_max and R_max
deployments = [
    {
        "model_id": "qwen-short-big-gpu",
        "description": "Custom Tuning",
        "config": {
            "gpu": "ergos-06-nv:gpu0",

            "model_name": "Qwen/Qwen3-0.6B-GPTQ-Int8",
            "vram_weights_gb": 0.763,

            "max_input_prompt_length": 1024,
            "max_output_prompt_length": 2048,

            "max_num_seqs": 850,
            "max_num_batched_tokens": 16384,
            "swap_space_per_instance": 0,
        }
    },
    # {
    #     "model_id": "qwen-short-small-gpu",
    #     "description": "Custom Tuning",
    #     "config": {
    #         "gpu": "ergos-02-nv:gpu0",

    #         "model_name": "Qwen/Qwen3-0.6B-GPTQ-Int8",
    #         "vram_weights_gb": 0.763,

    #         "max_input_prompt_length": 1024,
    #         "max_output_prompt_length": 2048,

    #         "max_num_seqs": 850,
    #         "max_num_batched_tokens": 16384,
    #         "swap_space_per_instance": 0,
    #     }
    # },
    # {
    #     "model_id": "qwen-short-small-gpu-again",
    #     "description": "Custom Tuning",
    #     "config": {
    #         "gpu": "ergos-02-nv:gpu0",

    #         "model_name": "Qwen/Qwen3-0.6B-GPTQ-Int8",
    #         "vram_weights_gb": 0.763,

    #         "max_input_prompt_length": 1024,
    #         "max_output_prompt_length": 2048,

    #         "max_num_seqs": 200,
    #         "max_num_batched_tokens": 4096,
    #         "swap_space_per_instance": 0,
    #     }
    # },
    # {
    #     "model_id": "qwen-long-big-gpu",
    #     "description": "Long variant - same config as short",
    #     "config": {
    #         "gpu": "ergos-06-nv:gpu0",

    #         "model_name": "Qwen/Qwen3-0.6B-GPTQ-Int8",
    #         "vram_weights_gb": 0.763,

    #         "max_input_prompt_length": 4096,
    #         "max_output_prompt_length": 6144,

    #         "max_num_seqs": 850,
    #         "max_num_batched_tokens": 16384,
    #         "swap_space_per_instance": 0,
    #     }
    # },
    # {
    #     "model_id": "qwen-long-small-gpu",
    #     "description": "Long variant - same config as short",
    #     "config": {
    #         "gpu": "ergos-02-nv:gpu0",

    #         "model_name": "Qwen/Qwen3-0.6B-GPTQ-Int8",
    #         "vram_weights_gb": 0.763,

    #         "max_input_prompt_length": 4096,
    #         "max_output_prompt_length": 6144,

    #         "max_num_seqs": 850,
    #         "max_num_batched_tokens": 16384,
    #         "swap_space_per_instance": 0,
    #     }
    # },
    # {
    #     "model_id": "qwen-long-small-gpu-again",
    #     "description": "Long variant - same config as short",
    #     "config": {
    #         "gpu": "ergos-02-nv:gpu0",

    #         "model_name": "Qwen/Qwen3-0.6B-GPTQ-Int8",
    #         "vram_weights_gb": 0.763,

    #         "max_input_prompt_length": 4096,
    #         "max_output_prompt_length": 6144,

    #         "max_num_seqs": 200,
    #         "max_num_batched_tokens": 4096,
    #         "swap_space_per_instance": 0,
    #     }
    # },
]





# Test each deployment
for idx, deployment in enumerate(deployments):
    model_id = deployment["model_id"]
    description = deployment["description"]
    config = deployment["config"]
    
    scheduler.deploy_model(model_id=model_id, **config)
    time.sleep(2)
    
    # Get calculation details
    serve_status = serve.status()
    max_num_seqs = None
    max_num_batched_tokens = None
    swap_space_gb = None
    for app_name, app in serve_status.applications.items():
        if app_name.startswith(f"{model_id}-") and not app_name.endswith("-router"):
            deployments = app.deployments if hasattr(app, 'deployments') else app.get('deployments', {})
            for deployment_name in deployments.keys():
                if deployment_name.startswith(f"{model_id}-") and not deployment_name.endswith("-router"):
                    handle = serve.get_deployment_handle(deployment_name, app_name=app_name)
                    calc_details = handle.get_calculation_details.remote().result()
                    max_num_seqs = calc_details.get('max_num_seqs')
                    max_num_batched_tokens = calc_details.get('max_num_batched_tokens')
                    swap_space_gb = calc_details.get('swap_space_gb')
                    break
            break
    
    prompt = "Write a short poem about beer"
    amount = 10_000
    prompts = [f"{prompt} {i}" for i in range(amount)]
    
    _ = inference_batch(prompts[:10], model_id=model_id, max_tokens=100, temperature=0.0)
    time.sleep(2)
    
    start = time.time()
    results = inference_batch(prompts, model_id=model_id, max_tokens=100, temperature=0.0)
    elapsed = time.time() - start
    
    if results and len(results) == len(prompts):
        throughput = len(results) / elapsed
        swap_display = f"{swap_space_gb:.1f} GB" if swap_space_gb is not None and swap_space_gb > 0 else "0 GB (disabled)"
        print(f"Processed {len(results)} prompts in {elapsed:.3f}s ({throughput:.2f} req/s)")
        print(f"  max_num_seqs: {max_num_seqs}, max_num_batched_tokens: {max_num_batched_tokens}, swap_space: {swap_display}")
    
    scheduler.shutdown(model_id)
    
    if idx < len(deployments) - 1:
        time.sleep(3)

