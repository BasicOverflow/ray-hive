"""Print VRAM for currently deployed model."""
import json
import os

import ray
from ray_hive.core.gpu_registry import get_gpu_registry

ray.init(address="auto", namespace=os.environ.get("RAY_HIVE_NAMESPACE", "ray_hive"))
reg = get_gpu_registry()
models = ray.get(reg.list_model_ids.remote())
print("MODELS", models)
for mid in models:
    d = ray.get(reg.get_deployment.remote(mid))
    print("DEPLOYMENT", json.dumps(d, indent=2, default=str)[:2000])
gpus = ray.get(reg.get_all_gpus.remote())
for k, v in (gpus or {}).items():
    active = (v or {}).get("active") or {}
    if active:
        print(k, "total", v.get("total"), "free", v.get("free"), "active", active)
