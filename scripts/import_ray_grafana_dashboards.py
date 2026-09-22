#!/usr/bin/env python3
"""Import Ray Grafana dashboards and print datasource name."""
from __future__ import annotations

import json
import pathlib
import urllib.request
from base64 import b64encode

GRAFANA = "http://127.0.0.1:3000"
AUTH = b64encode(b"admin:iamroot6025").decode()
DASH_DIR = pathlib.Path("/tmp/ray-grafana-dashboards")


def req(method: str, path: str, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(
        GRAFANA + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Basic {AUTH}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(r, timeout=60) as resp:
        return json.load(resp)


def main() -> None:
    ds = req("GET", "/api/datasources")
    print("datasources", [(d["name"], d["uid"], d["type"]) for d in ds])
    uid = next(d["uid"] for d in ds if d["type"] == "prometheus")
    name = next(d["name"] for d in ds if d["type"] == "prometheus")

    files = sorted(DASH_DIR.glob("*.json"))
    print("importing", len(files), "dashboards")
    for f in files:
        dash = json.loads(f.read_text(encoding="utf-8"))
        # Normalize datasource references to local prometheus
        raw = json.dumps(dash)
        raw = raw.replace('"Prometheus"', f'"{name}"')
        raw = raw.replace('"prometheus"', f'"{name}"')
        dash = json.loads(raw)
        # Ensure folder
        payload = {
            "dashboard": dash.get("dashboard", dash),
            "overwrite": True,
            "message": "Import Ray default dashboard",
        }
        # If top-level is already dashboard object with title
        if "title" in payload["dashboard"] or "panels" in payload["dashboard"]:
            pass
        elif "dashboard" in dash:
            payload["dashboard"] = dash["dashboard"]
        try:
            out = req("POST", "/api/dashboards/db", payload)
            print("OK", f.name, out.get("url") or out.get("status"))
        except Exception as e:
            print("FAIL", f.name, e)


if __name__ == "__main__":
    main()
