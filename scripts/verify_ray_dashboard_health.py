#!/usr/bin/env python3
import json, urllib.request

def get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read().decode()
    except Exception as e:
        return None, str(e)

for path in ["/api/prometheus_health", "/api/grafana_health"]:
    code, body = get("http://10.0.1.52:8265" + path)
    print(path, code)
    print(body[:500])
    print("---")

# insights has ray metrics with JobId?
q = "http://127.0.0.1:9090/api/v1/query?query=count%20by%20(JobId)%20(ray_actors)"
print("actors by JobId", urllib.request.urlopen(q, timeout=10).read().decode()[:800])
