import os, time, ray
from ray import serve

ns = os.environ.get("RAY_HIVE_NAMESPACE", "ray_hive")
if not ray.is_initialized():
    ray.init(address="auto", namespace=ns)

# Only kill deploy_service so next job picks up new code_rev; keep gpu_registry warm.
try:
    ray.kill(ray.get_actor("deploy_service", namespace=ns))
    print("killed deploy_service")
except Exception as e:
    print("kill skip deploy_service", type(e).__name__)

apps = serve.status().applications or {}
print("apps before:", list(apps))
for name in list(apps):
    if "qwen" in name.lower():
        print("deleting", name)
        try:
            serve.delete(name=name)
        except Exception as e:
            print("delete err", name, e)

# Clear registry entry for qwen if present
try:
    from ray_hive.core.gpu_registry import get_gpu_registry
    from ray_hive.core.ray_utils.lifecycle import shutdown_model
    try:
        shutdown_model("qwen25-vl-3b")
        print("shutdown_model ok")
    except Exception as e:
        print("shutdown_model", type(e).__name__, e)
    reg = get_gpu_registry()
    gpus = ray.get(reg.get_all_gpus.remote())
    print("registry gpus", len(gpus), list(gpus)[:8])
except Exception as e:
    print("registry", type(e).__name__, e)

time.sleep(2)
print("apps after:", list((serve.status().applications or {}).keys()))
print("CLEANUP_OK")
