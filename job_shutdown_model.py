"""Shutdown a model via Ray Job (avoids Windows Ray Client InProgressSentinel)."""
import os

import ray
from ray_hive import RayHive

if not ray.is_initialized():
    ray.init(address="auto", namespace=os.environ.get("RAY_HIVE_NAMESPACE", "ray_hive"))

hive = RayHive(address="auto", suppress_logging=False, show_banner=False)
mid = os.environ["NINI_MODEL_ID"]
try:
    hive.shutdown(mid)
    print("SHUTDOWN_OK", mid)
except Exception as e:
    print("shutdown warn", e)
