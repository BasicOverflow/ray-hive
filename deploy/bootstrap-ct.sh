#!/usr/bin/env bash
# Run inside the ray-hive supervisor LXC.
set -euo pipefail

: "${APP_ROOT:?}"
: "${APP_USER:?}"
: "${STATE_DIR:?}"
: "${ENV_FILE:?}"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl ca-certificates

if ! id -u "$APP_USER" >/dev/null 2>&1; then
  useradd -r -m -d /home/"$APP_USER" -s /bin/bash "$APP_USER"
fi

mkdir -p "$STATE_DIR" "$APP_ROOT"
chown -R "$APP_USER:$APP_USER" "$STATE_DIR" "$APP_ROOT"

if ! grep -q '^RAY_ADDRESS=' "$ENV_FILE" 2>/dev/null; then
  echo 'RAY_ADDRESS=ray://10.0.1.52:10001' >>"$ENV_FILE"
fi
if ! grep -q '^RAY_HIVE_NAMESPACE=' "$ENV_FILE" 2>/dev/null; then
  echo 'RAY_HIVE_NAMESPACE=ray_hive' >>"$ENV_FILE"
fi
if ! grep -q '^HIVE_SUPERVISOR_STATE=' "$ENV_FILE" 2>/dev/null; then
  echo "HIVE_SUPERVISOR_STATE=${STATE_DIR}/state.db" >>"$ENV_FILE"
fi
chown "$APP_USER:$APP_USER" "$ENV_FILE"

# Match Ray head Python (3.12) via uv — Ray Client rejects version skew.
su -s /bin/bash "$APP_USER" -c "
  set -e
  cd '$APP_ROOT'
  export PATH=\"\$HOME/.local/bin:\$PATH\"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=\"\$HOME/.local/bin:\$PATH\"
  fi
  uv python install 3.12
  uv venv --clear --python 3.12 .venv
  uv pip install --python .venv/bin/python -U pip wheel
  uv pip install --python .venv/bin/python \
    'ray[default,serve]==2.56.0' 'pydantic>=2.0.0' \
    fastapi==0.133.0 starlette==1.0.1 uvicorn rich
  uv pip install --python .venv/bin/python -e . --no-deps
"

install -m 0644 "$APP_ROOT/deploy/ray-hive-supervisor.service" /etc/systemd/system/ray-hive-supervisor.service
systemctl daemon-reload
systemctl enable --now ray-hive-supervisor
systemctl --no-pager --full status ray-hive-supervisor || true
