"""GPU placement envelope for train / custom infer — not vLLM Serve."""
from __future__ import annotations

from typing import Any, Callable, TypeVar

import ray

from ray_hive.jobs.envelopes import WORKER_LABEL_SELECTOR, _merge

F = TypeVar("F", bound=Callable[..., Any])


def gpu_options(num_gpus: float = 1.0, **kw: Any) -> dict:
    return _merge(
        {
            "scheduling_strategy": "SPREAD",
            "label_selector": WORKER_LABEL_SELECTOR,
            "num_gpus": num_gpus,
        },
        kw,
    )


def gpu_remote(num_gpus: float = 1.0, **kw: Any) -> Callable[[F], F]:
    """Place task/actor on GPU workers; always avoids head + vram-monitor pods."""
    return ray.remote(**gpu_options(num_gpus=num_gpus, **kw))
