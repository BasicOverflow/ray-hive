"""Poll live registry VRAM before / between live Deploy steps."""
from __future__ import annotations

import time
from typing import Callable


def gpu_available_gb(state: dict, gpu_key: str) -> float:
    info = state.get(gpu_key) or {}
    return float(info.get("available", info.get("free", 0.0)))


def wait_for_available(
    get_vram_state: Callable[[], dict],
    gpu_key: str,
    min_gb: float,
    *,
    poll_s: float = 2.0,
    max_wait_s: float = 120.0,
) -> float:
    """Block until gpu_key reports available >= min_gb. Raises TimeoutError."""
    deadline = time.time() + max_wait_s
    last = 0.0
    while True:
        state = get_vram_state()
        last = gpu_available_gb(state, gpu_key)
        if last >= min_gb:
            return last
        if time.time() >= deadline:
            raise TimeoutError(
                f"{gpu_key} still has {last:.2f}GB available "
                f"(need >={min_gb:.2f}GB); registry={ {k: round(gpu_available_gb(state, k), 2) for k in state} }"
            )
        time.sleep(poll_s)
