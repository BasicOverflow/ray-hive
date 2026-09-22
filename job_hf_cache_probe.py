"""List HF hub caches on every GPU worker (for prune planning)."""
import json
import os
from pathlib import Path

import ray


@ray.remote(num_gpus=0.01)
def probe_hf():
    import socket

    host = socket.gethostname()
    roots = []
    for p in (
        Path.home() / ".cache" / "huggingface",
        Path("/home/ray/.cache/huggingface"),
        Path(os.environ.get("HF_HOME") or ""),
        Path(os.environ.get("HUGGINGFACE_HUB_CACHE") or ""),
        Path("/tmp/huggingface"),
        Path("/mnt/huggingface"),
        Path("/data/huggingface"),
    ):
        if p and str(p) not in ("", ".") and p.exists():
            roots.append(p)

    # also search common parents
    for base in (Path("/home/ray"), Path("/tmp"), Path("/mnt"), Path("/data")):
        if not base.exists():
            continue
        for hit in base.rglob("models--*"):
            # limit depth noise
            if len(hit.parts) - len(base.parts) > 6:
                continue
            roots.append(hit.parent if hit.parent.name == "hub" else hit.parent)

    uniq = []
    seen = set()
    for r in roots:
        key = str(r.resolve()) if r.exists() else str(r)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(Path(key))

    models = []
    for root in uniq:
        hub = root / "hub" if (root / "hub").is_dir() else root
        if not hub.is_dir():
            continue
        for child in sorted(hub.iterdir()):
            if not child.name.startswith("models--"):
                continue
            try:
                sz = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
            except OSError:
                continue
            models.append(
                {
                    "path": str(child),
                    "name": child.name,
                    "gb": round(sz / 1e9, 3),
                }
            )
    return {
        "host": host,
        "HF_HOME": os.environ.get("HF_HOME"),
        "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "models": models,
        "total_gb": round(sum(m["gb"] for m in models), 3),
    }


def main():
    ray.init(address="auto", namespace=os.environ.get("RAY_HIVE_NAMESPACE", "ray_hive"))
    # one task per GPU present (rough) — pull enough to cover nodes
    n_gpu = int(ray.cluster_resources().get("GPU", 0))
    refs = [probe_hf.remote() for _ in range(max(n_gpu, 1) * 2)]
    rows = []
    pending = refs
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=30)
        for r in ready:
            try:
                rows.append(ray.get(r))
            except Exception as e:
                rows.append({"error": str(e)})
    # unique by host
    by_host = {}
    for row in rows:
        h = row.get("host") or row.get("error")
        if h not in by_host or (row.get("total_gb") or 0) > (by_host[h].get("total_gb") or 0):
            by_host[h] = row
    print(json.dumps(list(by_host.values()), indent=2))


if __name__ == "__main__":
    main()
