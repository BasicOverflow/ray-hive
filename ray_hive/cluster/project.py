"""Per-project identity on a shared Ray cluster."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SYSTEM_NAMESPACE = "ray_hive_system"


@dataclass
class ProjectContext:
    """Namespace, Serve route, model-id, and scratch prefixes for one prod system."""

    project_id: str
    namespace: str
    serve_route_prefix: str | None = None
    model_id_prefix: str | None = None
    scratch_prefix: str | None = None
    heartbeat_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.project_id or not self.namespace:
            raise ValueError("project_id and namespace are required")
        if self.serve_route_prefix and not self.serve_route_prefix.startswith("/"):
            self.serve_route_prefix = "/" + self.serve_route_prefix
        if self.scratch_prefix is None:
            self.scratch_prefix = f"/tmp/{self.project_id}-"
        if self.model_id_prefix is None:
            self.model_id_prefix = f"{self.project_id}-"
        if self.heartbeat_id is None:
            self.heartbeat_id = self.project_id

    def claim_model_id(self, suffix: str) -> str:
        return f"{self.model_id_prefix}{suffix}"
