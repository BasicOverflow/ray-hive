import os, ray, time
ns = os.environ.get("RAY_HIVE_NAMESPACE", "ray_hive")
if not ray.is_initialized():
    ray.init(address="auto", namespace=ns)
print("import begin", flush=True)
t0=time.time()
from ray_hive.core import deployment as dep
print("import done", time.time()-t0, "rev", dep.HIVE_DEPLOY_CODE_REV, flush=True)
print("avail", {k:v for k,v in ray.available_resources().items() if k in ('CPU','GPU','ergos-04-nv_gpu0') or 'ergos-04' in k}, flush=True)
try:
    from ray.util.state import list_tasks, list_actors
    for t in list_tasks():
        name=t.get("name") or ""
        if "deploy" in name.lower() or "router" in name.lower() or "DeployService" in name:
            print("TASK", name, t.get("state"), t.get("required_resources"))
    for a in list_actors():
        name=a.get("name") or a.get("actor_id") or ""
        if "deploy" in str(name).lower() or "router" in str(name).lower() or "qwen" in str(name).lower() or "gpu_registry" in str(name).lower():
            print("ACTOR", name, a.get("state"), a.get("repr_or_name"))
except Exception as e:
    print("state err", e)
