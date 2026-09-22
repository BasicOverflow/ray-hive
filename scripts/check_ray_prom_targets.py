#!/usr/bin/env python3
import json, urllib.request
d = json.load(urllib.request.urlopen("http://10.0.1.53:9090/api/v1/targets", timeout=10))
ts = d["data"]["activeTargets"]
print("targets", len(ts))
for t in ts:
    print(t.get("health"), (t.get("labels") or {}).get("instance"), (t.get("lastError") or "")[:100])
