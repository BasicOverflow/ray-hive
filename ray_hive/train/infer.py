"""Custom module Serve/actor infer — not vLLM."""
from __future__ import annotations

import logging
from typing import Any, Callable

from ray_hive.cluster.registry import ClaimKind, get_cluster_registry

logger = logging.getLogger("ray_hive.train.infer")


class CustomInferService:
    """Register a model_id claim and run a user deploy callable (Serve or actors).

    Sticky pins go through the same DesiredState path as vLLM when the meta
    supervisor is configured; the runner itself is project-supplied.
    """

    def __init__(
        self,
        model_id: str,
        project_id: str,
        *,
        deploy_fn: Callable[[], Any],
        shutdown_fn: Callable[[], Any] | None = None,
    ):
        self.model_id = model_id
        self.project_id = project_id
        self.deploy_fn = deploy_fn
        self.shutdown_fn = shutdown_fn

    def deploy(self) -> Any:
        import ray

        reg = get_cluster_registry()
        ray.get(
            reg.claim.remote(
                ClaimKind.MODEL_ID.value,
                self.model_id,
                self.project_id,
                meta={"kind": "custom_infer"},
            )
        )
        logger.info("custom infer deploy model_id=%s project=%s", self.model_id, self.project_id)
        return self.deploy_fn()

    def shutdown(self) -> None:
        import ray

        if self.shutdown_fn:
            self.shutdown_fn()
        try:
            reg = get_cluster_registry()
            ray.get(reg.release.remote(ClaimKind.MODEL_ID.value, self.model_id, self.project_id))
        except Exception:
            pass
