"""Serve app heal with cool-down + route claim."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from ray_hive.cluster.registry import ClaimKind, get_cluster_registry

logger = logging.getLogger("ray_hive.jobs.serve_heal")


class ServeAppHeal:
    def __init__(
        self,
        project_id: str,
        route_prefix: str,
        *,
        deploy_fn: Callable[[], Any],
        health_fn: Callable[[], bool] | None = None,
        cooldown_s: float = 600.0,
        claim_route: bool = True,
    ):
        self.project_id = project_id
        self.route_prefix = route_prefix if route_prefix.startswith("/") else f"/{route_prefix}"
        self.deploy_fn = deploy_fn
        self.health_fn = health_fn
        self.cooldown_s = cooldown_s
        self._last_attempt = 0.0
        if claim_route:
            self._claim()

    def _claim(self) -> None:
        import ray

        reg = get_cluster_registry()
        ray.get(
            reg.claim.remote(
                ClaimKind.SERVE_ROUTE.value,
                self.route_prefix,
                self.project_id,
            )
        )

    def should_attempt(self, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        return (now - self._last_attempt) >= self.cooldown_s

    def heal_if_needed(self, *, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and not self.should_attempt(now):
            return False
        healthy = True
        if self.health_fn is not None:
            try:
                healthy = bool(self.health_fn())
            except Exception:
                healthy = False
        if healthy and not force:
            return False
        self._last_attempt = now
        logger.info("serve heal deploy route=%s project=%s", self.route_prefix, self.project_id)
        self.deploy_fn()
        return True
