"""Prune leftover HuggingFace model caches on all GPU Ray workers.

Keeps models whose hub dir name matches KEEP_SUBSTR (comma-separated env).
Default keep: Qwen2.5-VL-3B (production OCR pick). Deletes everything else
under HF hub, then drops unreferenced blobs.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import ray

KEEP_DEFAULT = "Qwen2.5-VL-3B-Instruct,Qwen2.5-VL-3B-Instruct-AWQ"


@ray.remote(num_gpus=0.01)
def prune_hf(keep_substr: list[str], dry_run: bool = False) -> dict:
    import socket

    host = socket.gethostname()
    hub = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface")) / "hub"
    if not hub.is_dir():
        hub = Path.home() / ".cache" / "huggingface" / "hub"
    if not hub.is_dir():
        return {"host": host, "hub": str(hub), "skipped": "no hub", "deleted": [], "kept": []}

    deleted = []
    kept = []
    for child in sorted(hub.iterdir()):
        if not child.name.startswith("models--"):
            continue
        keep = any(s.lower() in child.name.lower() for s in keep_substr if s)
        try:
            # follow symlinks into blobs for size
            sz = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
        except OSError:
            sz = 0
        gb = round(sz / 1e9, 3)
        entry = {"name": child.name, "gb": gb, "path": str(child)}
        if keep:
            kept.append(entry)
            continue
        deleted.append(entry)
        if not dry_run:
            shutil.rmtree(child, ignore_errors=True)

    # drop lock leftovers + unreferenced blobs (best-effort)
    blobs = hub / "blobs"
    if not dry_run and deleted:
        locks = hub / ".locks"
        if locks.is_dir():
            for lock_dir in list(locks.iterdir()):
                if lock_dir.is_dir() and not any(lock_dir.iterdir()):
                    shutil.rmtree(lock_dir, ignore_errors=True)
        # If nothing kept, wipe blobs entirely (emptyDir leftover from trials).
        if not kept and blobs.is_dir():
            shutil.rmtree(blobs, ignore_errors=True)
            blobs.mkdir(parents=True, exist_ok=True)
        elif blobs.is_dir():
            referenced: set[Path] = set()
            for snap in hub.glob("models--*/snapshots/*/*"):
                try:
                    if snap.is_symlink() or snap.is_file():
                        referenced.add(snap.resolve())
                except OSError:
                    pass
            for blob in list(blobs.iterdir()):
                if blob.is_file():
                    try:
                        if blob.resolve() not in referenced:
                            blob.unlink()
                    except OSError:
                        pass

    # report remaining size
    total = 0
    if hub.is_dir():
        try:
            total = sum(f.stat().st_size for f in hub.rglob("*") if f.is_file())
        except OSError:
            pass
    return {
        "host": host,
        "hub": str(hub),
        "dry_run": dry_run,
        "deleted": deleted,
        "kept": kept,
        "hub_gb_after": round(total / 1e9, 3),
    }


def main() -> None:
    import json

    keep = [
        s.strip()
        for s in (os.environ.get("NINI_HF_KEEP") or KEEP_DEFAULT).split(",")
        if s.strip()
    ]
    dry = (os.environ.get("NINI_HF_PRUNE_DRY") or "").strip() in ("1", "true", "yes")
    ray.init(address="auto", namespace=os.environ.get("RAY_HIVE_NAMESPACE", "ray_hive"))
    n_gpu = int(ray.cluster_resources().get("GPU", 0))
    refs = [prune_hf.remote(keep, dry) for _ in range(max(n_gpu * 2, 2))]
    rows = []
    pending = list(refs)
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=60)
        for r in ready:
            try:
                rows.append(ray.get(r))
            except Exception as e:
                rows.append({"error": str(e)})
    by_host = {}
    for row in rows:
        h = row.get("host") or "err"
        if h not in by_host:
            by_host[h] = row
    print(json.dumps({"keep": keep, "hosts": list(by_host.values())}, indent=2))


if __name__ == "__main__":
    main()
