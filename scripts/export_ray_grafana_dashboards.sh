#!/bin/bash
set -euo pipefail
HEAD=$(sudo k3s kubectl get pod -n ray-system -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
mkdir -p /tmp/ray-grafana-dashboards
sudo k3s kubectl cp -n ray-system "$HEAD:/tmp/ray/session_latest/metrics/grafana/dashboards" /tmp/ray-grafana-dashboards/
ls -la /tmp/ray-grafana-dashboards
