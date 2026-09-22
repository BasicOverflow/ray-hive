#!/bin/bash
set -e
cp -a /mnt/monitoring/configs/prometheus.yml /mnt/monitoring/configs/prometheus.yml.bak.ray
python3 /tmp/add_ray_prom_scrape.py
curl -sS -X POST http://127.0.0.1:9090/-/reload
sleep 3
curl -sS 'http://127.0.0.1:9090/api/v1/query?query=up%7Bjob%3D%22ray%22%7D' | python3 -m json.tool | head -50
curl -sS 'http://127.0.0.1:9090/api/v1/label/__name__/values' | python3 -c 'import sys,json; n=[x for x in json.load(sys.stdin)["data"] if x.startswith("ray_")]; print("ray_metrics", len(n)); print("\n".join(n[:30]))'
