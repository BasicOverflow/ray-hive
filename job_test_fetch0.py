import os, ray, time
ns = os.environ.get("RAY_HIVE_NAMESPACE", "ray_hive")
if not ray.is_initialized():
    ray.init(address="auto", namespace=ns)
from ray_hive.core.deployment import fetch_hf_config_dict
res = "ergos-04-nv_gpu0"
print("submit", res, "num_cpus=0", flush=True)
t0=time.time()
fut = fetch_hf_config_dict.options(resources={res: 0.01}, num_cpus=0).remote("Qwen/Qwen2.5-VL-3B-Instruct")
cfg = ray.get(fut, timeout=120)
print("OK", time.time()-t0, "keys", list(cfg)[:6] if isinstance(cfg, dict) else type(cfg))
