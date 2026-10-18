"""Run one ops task per worker node."""
from __future__ import annotations

import logging
from typing import Any, Callable

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from ray_hive.jobs.envelopes import ops_options

logger = logging.getLogger("ray_hive.jobs.node_broadcast")


class NodeBroadcast:
    """Place ``fn`` once on each alive non-head worker node."""

    def __init__(self, *, skip_headgroup: bool = True):
        self.skip_headgroup = skip_headgroup

    def _node_ids(self) -> list[str]:
        nodes = ray.nodes()
        out = []
        for n in nodes:
            if not n.get("Alive"):
                continue
            nid = n.get("NodeID")
            if not nid:
                continue
            resources = n.get("Resources") or {}
            # Heuristic: head often has fewer custom labels; prefer nodes with CPU
            if resources.get("CPU", 0) <= 0:
                continue
            out.append(nid)
        return out

    def run(self, remote_fn, *args, **kwargs) -> list[Any]:
        """``remote_fn`` must be a Ray remote function (ops envelope recommended)."""
        refs = []
        for nid in self._node_ids():
            strat = NodeAffinitySchedulingStrategy(node_id=nid, soft=False)
            refs.append(
                remote_fn.options(scheduling_strategy=strat, **ops_options()).remote(
                    *args, **kwargs
                )
            )
        logger.info("node_broadcast nodes=%s", len(refs))
        return ray.get(refs) if refs else []
