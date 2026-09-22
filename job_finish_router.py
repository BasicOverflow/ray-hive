import os, ray, time
from ray import serve

ns = os.environ.get("RAY_HIVE_NAMESPACE", "ray_hive")
if not ray.is_initialized():
    ray.init(address="auto", namespace=ns)

apps = serve.status().applications or {}
print("APPS", list(apps))
for name, app in apps.items():
    print(f"APP {name} status={app.status}")
    for dep_name, dep in (app.deployments or {}).items():
        print(f"  DEP {dep_name} status={dep.status} msg={dep.message}")

print("AVAILABLE_CPU", ray.available_resources().get("CPU"))
print("starting deploy_router...")
from ray_hive.core.deployment import deploy_router

# metadata stub - plan dict used by router for model card; can be minimal
replica_id = "qwen25-vl-3b-ergos-04-nv-gpu0"
resource = "ergos-04-nv_gpu0"
meta = {replica_id: {"total_vram_gb": 11.7, "max_model_len": 11520, "max_num_seqs": 4, "max_num_batched_tokens": 2496, "gpu_memory_utilization": 0.73}}

fut = deploy_router.options(resources={resource: 0.01}, num_cpus=0.25).remote(
    "qwen25-vl-3b",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    [replica_id],
    meta,
    resource,
    {},
    -1,
    -1,
    False,
    True,
    None,
)
print("future submitted, waiting...")
try:
    print("result", ray.get(fut, timeout=300))
except Exception as e:
    print("FAIL", type(e).__name__, e)
    apps = serve.status().applications or {}
    app = apps.get("qwen25-vl-3b")
    if app:
        print("router app", app.status)
        for dep_name, dep in (app.deployments or {}).items():
            print(f"  {dep_name}: {dep.status} — {dep.message}")
print("APPS after", list((serve.status().applications or {}).keys()))
