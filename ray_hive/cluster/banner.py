"""One-shot stdout banner with GitHub URL."""
from __future__ import annotations

import sys

_PRINTED = False
GITHUB_URL = "https://github.com/BasicOverflow/ray-hive"


def print_hive_banner(*, force: bool = False) -> None:
    """Print ``ray-hive <ver> — <github>`` once per process (GitHub only)."""
    global _PRINTED
    if _PRINTED and not force:
        return
    try:
        from importlib.metadata import version

        ver = version("ray-hive")
    except Exception:
        ver = getattr(__import__("ray_hive"), "__version__", "unknown")
    print(f"ray-hive {ver} — {GITHUB_URL}", file=sys.stdout, flush=True)
    _PRINTED = True
