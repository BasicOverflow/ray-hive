import os, ray, time
ns = os.environ.get("RAY_HIVE_NAMESPACE", "ray_hive")
if not ray.is_initialized():
    ray.init(address="auto", namespace=ns)

print("node", ray.get_runtime_context().get_node_id())
print("avail cpu", ray.available_resources().get("CPU"))
print("avail gpu keys", {k:v for k,v in ray.available_resources().items() if "_gpu" in k})

# Can we import transformers here (job worker)?
try:
    import transformers
    print("transformers", transformers.__version__)
except Exception as e:
    print("no transformers", e)

from ray_hive.core.deployment import fetch_hf_config_dict, load_hf_config_dict

# Try direct load on this process
try:
    t0=time.time()
    cfg = load_hf_config_dict("Qwen/Qwen2.5-VL-3B-Instruct")
    print("direct load ok", time.time()-t0, list(cfg)[:8] if isinstance(cfg, dict) else type(cfg))
except Exception as e:
    print("direct load fail", type(e).__name__, e)

# Try remote with custom resource + timeout
res = "ergos-04-nv_gpu1"
print("submitting remote on", res, flush=True)
fut = fetch_hf_config_dict.options(resources={res: 0.01}, num_cpus=0.05).remote("Qwen/Qwen2.5-VL-3B-Instruct")
try:
    print("remote result", ray.get(fut, timeout=60))
except Exception as e:
    print("remote fail", type(e).__name__, e)
    # cancel
    try: ray.cancel(fut, force=True)
    except: pass

# Try remote with num_gpus
print("submitting remote num_gpus", flush=True)
fut2 = fetch_hf_config_dict.options(num_gpus=0.01, num_cpus=0.05).remote("Qwen/Qwen2.5-VL-3B-Instruct")
try:
    print("num_gpus result keys", list(ray.get(fut2, timeout=90))[:5] if isinstance(ray.get(fut2, timeout=1), dict) else ray.get(fut2, timeout=90))
except Exception as e:
    print("num_gpus fail", type(e).__name__, e)
    try: ray.cancel(fut2, force=True)
    except: pass
