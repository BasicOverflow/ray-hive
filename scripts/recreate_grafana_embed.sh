#!/bin/bash
# Recreate Grafana with embedding enabled for Ray Dashboard iframes.
set -euo pipefail
IMG=$(docker inspect grafana --format '{{.Config.Image}}')
NETWORK=$(docker inspect grafana --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}')
echo image=$IMG network=$NETWORK

docker stop grafana
docker rm grafana

docker run -d \
  --name grafana \
  --restart unless-stopped \
  --network "$NETWORK" \
  -p 3000:3000 \
  -e GF_SECURITY_ADMIN_USER=admin \
  -e GF_SECURITY_ADMIN_PASSWORD=iamroot6025 \
  -e GF_USERS_ALLOW_SIGN_UP=false \
  -e GF_SECURITY_ALLOW_EMBEDDING=true \
  -e GF_AUTH_ANONYMOUS_ENABLED=true \
  -e GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
  -e GF_SECURITY_COOKIE_SAMESITE=disabled \
  -v /mnt/monitoring/grafana:/var/lib/grafana \
  -v /mnt/monitoring/configs/grafana/provisioning:/etc/grafana/provisioning \
  "$IMG"

sleep 5
curl -sS -u admin:iamroot6025 http://127.0.0.1:3000/api/health
docker exec grafana grafana cli --version 2>/dev/null || true
grep -i allow_embedding /usr/share/grafana/conf/defaults.ini 2>/dev/null || docker exec grafana sh -c 'env | grep -i GF_'
