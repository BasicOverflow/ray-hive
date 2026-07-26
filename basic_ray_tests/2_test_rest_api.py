"""
Test Ray REST API endpoints.

Verifies:
- Ray dashboard is accessible
- Version, cluster status, nodes
- Job submission API (submit + status + logs)
- Ray Serve applications (deployed models)
"""
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / "examples" / ".env")

BASE = os.environ["RAY_DASHBOARD_URL"].rstrip("/")


def get_json(path, optional=False):
    resp = requests.get(f"{BASE}{path}", timeout=10)
    if optional and resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def post_json(path, payload, optional=False):
    resp = requests.post(f"{BASE}{path}", json=payload, timeout=10)
    if optional and resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def print_nodes(nodes_resp):
    summary = nodes_resp.get("data", {}).get("summary", [])
    if summary:
        alive = [n for n in summary if n.get("raylet", {}).get("state") == "ALIVE"]
        gpus = sum(len(n["gpus"]) if n.get("gpus") else n.get("resources", {}).get("GPU", 0) for n in alive)
        cpus = sum(n.get("resources", {}).get("CPU", 0) for n in alive)
        print(f"✅ Nodes: {len(alive)} alive, {cpus:.0f} CPUs, {gpus:.0f} GPUs")
        return

    nodes = nodes_resp.get("data", {}).get("nodes", [])
    alive = [n for n in nodes if n.get("alive", True)]
    gpus = sum(n.get("resources", {}).get("GPU", 0) for n in alive)
    cpus = sum(n.get("resources", {}).get("CPU", 0) for n in alive)
    print(f"✅ Nodes: {len(alive)} alive, {cpus:.0f} CPUs, {gpus:.0f} GPUs")


def main():
    print(f"Testing Ray dashboard at {BASE}...")

    version = get_json("/api/version")
    ray_version = version.get("ray_version") or version.get("version")
    print(f"✅ Ray version: {ray_version}")

    status = get_json("/api/cluster_status")
    print(f"✅ Cluster status: {status.get('result', status)}")

    nodes_resp = get_json("/api/nodes?view=summary", optional=True)
    if nodes_resp is None:
        nodes_resp = get_json("/api/nodes", optional=True)
    if nodes_resp is None:
        print("ℹ️  Nodes endpoint not available on this cluster")
    else:
        print_nodes(nodes_resp)

    jobs = get_json("/api/jobs/")
    print(f"✅ Jobs listed: {len(jobs)}")

    job_id = post_json("/api/jobs/", {
        "entrypoint": "python -c \"print('ray-hive rest api test')\"",
    })["job_id"]
    print(f"✅ Submitted test job: {job_id}")

    deadline = time.time() + 60
    state = None
    while time.time() < deadline:
        job = get_json(f"/api/jobs/{job_id}")
        state = job.get("status")
        if state in {"SUCCEEDED", "FAILED", "STOPPED"}:
            break
        time.sleep(1)

    logs = get_json(f"/api/jobs/{job_id}/logs").get("logs", "")
    print(f"✅ Job status: {state}")
    print(f"   logs: {logs.strip() or '(empty)'}")

    serve = get_json("/api/serve/applications/", optional=True)
    if serve is None:
        print("ℹ️  Serve applications endpoint not available")
    else:
        apps = serve.get("applications", {})
        print(f"✅ Serve applications: {len(apps)}")
        for app_name in sorted(apps):
            print(f"   - {app_name}")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
