#!/usr/bin/env python3
import json, urllib.request, urllib.parse

base = "http://10.0.1.53:9090"

def get(path, params=None):
    url = base + path
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    return json.load(urllib.request.urlopen(url, timeout=15))

print("healthy", urllib.request.urlopen(base + "/-/healthy", timeout=5).read())
targets = get("/api/v1/targets")["data"]["activeTargets"]
print("active", len(targets), "up", sum(1 for t in targets if t["health"]=="up"))
q = get("/api/v1/query", {"query": 'count({__name__=~"ray_.+"})'})
print("ray series count query", q["data"]["result"][:3])
names = [n for n in get("/api/v1/label/__name__/values")["data"] if n.startswith("ray_")]
print("ray metric names", len(names), names[:15])

# try federate
req = urllib.request.Request(
    base + "/federate?" + urllib.parse.urlencode([("match[]", '{__name__=~"ray_.+"}')])
)
body = urllib.request.urlopen(req, timeout=30).read().decode()
print("federate lines", len(body.splitlines()), "sample:")
print("\n".join(body.splitlines()[:15]))
