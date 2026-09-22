#!/usr/bin/env python3
"""Add Ray federation scrape job to insights prometheus.yml if missing."""
from pathlib import Path

p = Path("/mnt/monitoring/configs/prometheus.yml")
text = p.read_text()
marker = "10.0.1.53:9090"
if marker in text or 'job_name: "ray"' in text or "job_name: ray" in text:
    print("ray scrape already present")
else:
    block = """
  # Ray cluster metrics (via in-cluster Prometheus federation on k3s)
  - job_name: ray
    honor_labels: true
    metrics_path: /federate
    params:
      match[]:
        - '{job="ray"}'
        - '{__name__=~"ray_.+"}'
    static_configs:
      - targets: ["10.0.1.53:9090"]
        labels:
          cluster: ray-hive
"""
    if not text.endswith("\n"):
        text += "\n"
    p.write_text(text + block)
    print("added ray federation job")
