#!/usr/bin/env bash
# Run as root on home-media (Proxmox): create LXC + install ray-hive supervisor.
set -euo pipefail

REPO_CLONE_URL="${REPO_CLONE_URL:-http://10.0.0.52:3000/admin/ray-hive.git}"
WORK_DIR="${WORK_DIR:-/tmp/ray-hive-deploy}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing command: $1" >&2; exit 1; }
}

need_cmd pct
need_cmd curl

if [[ "$(id -u)" -ne 0 ]]; then
  echo "run as root on a Proxmox node (home-media)" >&2
  exit 1
fi

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

if ! command -v git >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq git
fi

echo "install: cloning ${REPO_CLONE_URL}"
git clone --depth 1 "$REPO_CLONE_URL" "$WORK_DIR"
find "$WORK_DIR" -type f -exec sed -i 's/\r$//' {} +

__OVERRIDES=$(mktemp)
for v in CTID CT_HOSTNAME CT_IP CT_GW CT_CIDR BRIDGE STORAGE TEMPLATE_STORAGE OSTEMPLATE MEMORY CORES DISK_GB UNPRIVILEGED FEATURES START_ON_BOOT HA_GROUP APP_ROOT APP_USER STATE_DIR ENV_FILE REPO_CLONE_URL; do
  if [[ -n "${!v+x}" ]]; then
    printf '%s=%q\n' "$v" "${!v}" >>"$__OVERRIDES"
  fi
done
# shellcheck disable=SC1091
source "${WORK_DIR}/deploy/defaults.env"
# shellcheck disable=SC1090
source "$__OVERRIDES"
rm -f "$__OVERRIDES"

echo "install: CTID=${CTID} hostname=${CT_HOSTNAME} ip=${CT_IP}/${CT_CIDR}"

if pct status "$CTID" >/dev/null 2>&1; then
  echo "install: CT ${CTID} exists — re-bootstrap"
  pct set "$CTID" --hostname "$CT_HOSTNAME" || true
else
  TEMPLATE_PATH="${TEMPLATE_STORAGE}:vztmpl/${OSTEMPLATE}"
  if ! pveam list "$TEMPLATE_STORAGE" 2>/dev/null | grep -q "$OSTEMPLATE"; then
    pveam update
    pveam download "$TEMPLATE_STORAGE" "$OSTEMPLATE"
  fi
  pct create "$CTID" "$TEMPLATE_PATH" \
    --hostname "$CT_HOSTNAME" \
    --cores "$CORES" \
    --memory "$MEMORY" \
    --swap 512 \
    --rootfs "${STORAGE}:${DISK_GB}" \
    --net0 "name=eth0,bridge=${BRIDGE},ip=${CT_IP}/${CT_CIDR},gw=${CT_GW}" \
    --unprivileged "$UNPRIVILEGED" \
    --features "$FEATURES" \
    --onboot "$START_ON_BOOT" \
    --start 1
  sleep 5
fi

pct start "$CTID" >/dev/null 2>&1 || true
for _ in $(seq 1 30); do
  pct exec "$CTID" -- true >/dev/null 2>&1 && break
  sleep 2
done

# Minimal env if missing (Ray MetalLB head)
pct exec "$CTID" -- bash -lc "mkdir -p $(dirname "$ENV_FILE") $STATE_DIR"
if ! pct exec "$CTID" -- test -f "$ENV_FILE"; then
  pct exec "$CTID" -- bash -lc "cat > $ENV_FILE <<'EOF'
RAY_ADDRESS=ray://10.0.1.52:10001
RAY_HIVE_NAMESPACE=ray_hive
HIVE_SUPERVISOR_STATE=${STATE_DIR}/state.db
EOF"
fi

echo "install: syncing app tree into CT ${APP_ROOT}"
pct exec "$CTID" -- bash -lc "rm -rf ${APP_ROOT} && mkdir -p ${APP_ROOT}"
tar -C "$WORK_DIR" -cf - \
  --exclude .git \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  . | pct exec "$CTID" -- tar -C "$APP_ROOT" -xf -

pct exec "$CTID" -- env \
  APP_ROOT="$APP_ROOT" \
  APP_USER="$APP_USER" \
  STATE_DIR="$STATE_DIR" \
  ENV_FILE="$ENV_FILE" \
  bash "${APP_ROOT}/deploy/bootstrap-ct.sh"

echo
echo "Done. CT ${CTID} (${CT_HOSTNAME}) @ ${CT_IP}"
echo "  logs: pct exec ${CTID} -- journalctl -u ray-hive-supervisor -f"
