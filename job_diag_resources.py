import os, json, ray
from ray import serve

ns = os.environ.get("RAY_HIVE_NAMESPACE", "ray_hive")
if not ray.is_initialized():
    ray.init(address="auto", namespace=ns)

print("APPS", list((serve.status().applications or {}).keys()))
print("CLUSTER_RESOURCES", {k:v for k,v in ray.cluster_resources().items() if 'ergos' in k.lower() or 'gpu' in k.lower() or 'nvidia' in k.lower() or k in ('CPU','GPU','memory')})
print("AVAILABLE", {k:v for k,v in ray.available_resources().items() if 'ergos' in k.lower() or 'gpu' in k.lower() or 'nvidia' in k.lower() or k in ('CPU','GPU')})

try:
    reg = ray.get_actor("gpu_registry", namespace=ns)
    print("REGISTRY_IDS", ray.get(reg.list_model_ids.remote()))
    for mid in ray.get(reg.list_model_ids.remote()):
        print("DEP", mid, ray.get(reg.get_deployment.remote(mid)))
    gpus = ray.get(reg.get_all_gpus.remote())
    for k,v in list(gpus.items())[:20]:
        print("GPU", k, {kk:v.get(kk) for kk in ('gpu_key','available_gb','total_gb','resource_name','hostname') if kk in v or True})
        print("  raw keys", list(v.keys())[:30])
except Exception as e:
    print("registry err", type(e).__name__, e)

# pending/infeasible via state API if available
try:
    from ray.util.state import list_tasks
    tasks = list(list_tasks(filters=[("state", "=", "PENDING_NODE_ASSIGNMENT")]))
    print("PENDING_TASKS", len(tasks))
    for t in tasks[:15]:
        print(" ", t.get("name"), t.get("required_resources"), t.get("state"))
except Exception as e:
    print("state api", type(e).__name__, e)
