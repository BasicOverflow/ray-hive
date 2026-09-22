#!/usr/bin/env python3
import json, urllib.request
d = json.load(urllib.request.urlopen("http://127.0.0.1:9090/api/v1/targets", timeout=10))
for t in d["data"]["activeTargets"]:
    job = (t.get("labels") or {}).get("job")
    if job and "ray" in job:
        print(job, t.get("health"), t.get("scrapeUrl"), (t.get("lastError") or "")[:200])
names = json.load(urllib.request.urlopen("http://127.0.0.1:9090/api/v1/label/__name__/values", timeout=10))["data"]
ray = [n for n in names if n.startswith("ray_")]
print("insights ray metrics", len(ray))
print(ray[:20])
