"""Cluster-wide rate gate as a named Ray actor."""
from __future__ import annotations

import asyncio
import time
from typing import Any

import ray

from ray_hive.jobs.envelopes import io_options


@ray.remote(max_concurrency=1, **io_options())
class RateGate:
    """Singleton pace limiter. Workers call ``acquire`` before external API hits."""

    def __init__(
        self,
        min_interval: float = 1.0,
        max_interval: float = 30.0,
        backoff_factor: float = 2.0,
        recovery_step: float = 0.1,
        knobs: dict | None = None,
    ):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.backoff_factor = backoff_factor
        self.recovery_step = recovery_step
        self._interval = min_interval
        self._last = 0.0
        self._knobs = dict(knobs or {})

    async def acquire(self) -> None:
        now = time.monotonic()
        wait = self._interval - (now - self._last)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last = time.monotonic()

    def record_success(self) -> None:
        self._interval = max(self.min_interval, self._interval - self.recovery_step)

    def record_rate_limited(self) -> None:
        self._interval = min(self.max_interval, self._interval * self.backoff_factor)

    def interval(self) -> float:
        return self._interval

    def get_knobs(self) -> dict:
        return dict(self._knobs)

    def set_knobs(self, knobs: dict) -> None:
        self._knobs.update(knobs)

    def ping(self) -> str:
        return "ok"


def get_or_create_rate_gate(
    name: str,
    *,
    namespace: str | None = None,
    min_interval: float = 1.0,
    knobs: dict | None = None,
    **actor_kw: Any,
):
    """Probe existing named gate or create detached."""
    ns = namespace
    try:
        actor = ray.get_actor(name, namespace=ns) if ns else ray.get_actor(name)
        ray.get(actor.ping.remote())
        return actor
    except Exception:
        pass
    opts = dict(
        name=name,
        lifetime="detached",
        max_restarts=-1,
        max_concurrency=1,
        **io_options(),
        **actor_kw,
    )
    if ns:
        opts["namespace"] = ns
    return RateGate.options(**opts).remote(
        min_interval=min_interval,
        knobs=knobs,
    )
