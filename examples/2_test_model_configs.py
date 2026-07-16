import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.inference import inference_batch

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

scheduler = RayHive(suppress_logging=True)


# Test configurations — planner estimates max_num_seqs / max_num_batched_tokens from prompt lengths
deployments = [
    {
        "model_id": "qwen-short-big-gpu",
        "description": "Default planner estimates",
        "config": {
            "gpu": "ergos-06-nv:gpu0",
            "model_name": "Qwen/Qwen3-0.6B-GPTQ-Int8",
            "max_input_prompt_length": 1024,
            "max_output_prompt_length": 2048,
            # "max_num_seqs": 24
        }
    },
    # {
    #     "model_id": "qwen-short-small-gpu",
    #     "description": "Override concurrency settings",
    #     "config": {
    #         "gpu": "ergos-02-nv:gpu0",
    #         "model_name": "Qwen/Qwen3-0.6B-GPTQ-Int8",
    #         "max_input_prompt_length": 1024,
    #         "max_output_prompt_length": 2048,
    #         "max_num_seqs": 850,
    #         "max_num_batched_tokens": 16384,
    #     }
    # },
    # {
    #     "model_id": "qwen-short-small-gpu-again",
    #     "description": "Lower concurrency override",
    #     "config": {
    #         "gpu": "ergos-02-nv:gpu0",
    #         "model_name": "Qwen/Qwen3-0.6B-GPTQ-Int8",
    #         "max_input_prompt_length": 1024,
    #         "max_output_prompt_length": 2048,
    #         "max_num_seqs": 200,
    #         "max_num_batched_tokens": 4096,
    #     }
    # },
    # {
    #     "model_id": "qwen-long-big-gpu",
    #     "description": "Long prompts — planner re-estimates from lengths",
    #     "config": {
    #         "gpu": "ergos-06-nv:gpu0",
    #         "model_name": "Qwen/Qwen3-0.6B-GPTQ-Int8",
    #         "max_input_prompt_length": 4096,
    #         "max_output_prompt_length": 6144,
    #         "max_num_seqs": 850,
    #         "max_num_batched_tokens": 16384,
    #     }
    # },
    # {
    #     "model_id": "qwen-long-small-gpu",
    #     "description": "Long prompts on smaller GPU",
    #     "config": {
    #         "gpu": "ergos-02-nv:gpu0",
    #         "model_name": "Qwen/Qwen3-0.6B-GPTQ-Int8",
    #         "max_input_prompt_length": 4096,
    #         "max_output_prompt_length": 6144,
    #         "max_num_seqs": 850,
    #         "max_num_batched_tokens": 16384,
    #     }
    # },
    # {
    #     "model_id": "qwen-long-small-gpu-again",
    #     "description": "Long prompts, lower concurrency",
    #     "config": {
    #         "gpu": "ergos-02-nv:gpu0",
    #         "model_name": "Qwen/Qwen3-0.6B-GPTQ-Int8",
    #         "max_input_prompt_length": 4096,
    #         "max_output_prompt_length": 6144,
    #         "max_num_seqs": 200,
    #         "max_num_batched_tokens": 4096,
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
        print(f"Processed {len(results)} prompts in {elapsed:.3f}s ({throughput:.2f} req/s)")
    
    scheduler.shutdown(model_id)
    
    if idx < len(deployments) - 1:
        time.sleep(3)

