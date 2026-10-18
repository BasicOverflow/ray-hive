"""@ray.remote option envelopes. Workload tasks always avoid head + vram-monitor pods."""
from __future__ import annotations

from typing import Any, Callable, TypeVar

import ray

# Workloads: worker node-group only. Head is headgroup; vram-monitor pods have no
# node-group and do not match — they stay infeasible. No public opt-out.
WORKER_LABEL_SELECTOR = {"ray.io/node-group": "!headgroup"}

IO_TASK_CPUS = 0.001

F = TypeVar("F", bound=Callable[..., Any])


def _merge(base: dict, kw: dict) -> dict:
    out = {**base, **kw}
    # Workload envelopes always force the worker label (callers cannot clear it).
    if base.get("label_selector") is not None:
        out["label_selector"] = dict(WORKER_LABEL_SELECTOR)
    return out


def io_options(**kw: Any) -> dict:
    return _merge(
        {
            "scheduling_strategy": "SPREAD",
            "label_selector": WORKER_LABEL_SELECTOR,
            "num_cpus": IO_TASK_CPUS,
        },
        kw,
    )


def cpu_options(num_cpus: float | int = 1, **kw: Any) -> dict:
    """``num_cpus=-1`` means AdmissionBroker should suggest; stored as sentinel."""
    opts = _merge(
        {
            "scheduling_strategy": "SPREAD",
            "label_selector": WORKER_LABEL_SELECTOR,
            "num_cpus": 1 if num_cpus == -1 else num_cpus,
        },
        kw,
    )
    if num_cpus == -1:
        opts["_broker_cpus"] = True
    return opts


def ops_options(**kw: Any) -> dict:
    """Head / every-node ops. No worker label (but never targets vram-monitor CPUs)."""
    return {"scheduling_strategy": "SPREAD", **kw}


def io_remote(**kw: Any) -> Callable[[F], F]:
    return ray.remote(**io_options(**kw))


def cpu_remote(num_cpus: float | int = 1, **kw: Any) -> Callable[[F], F]:
    return ray.remote(**cpu_options(num_cpus=num_cpus, **kw))


def ops_remote(**kw: Any) -> Callable[[F], F]:
    return ray.remote(**ops_options(**kw))
