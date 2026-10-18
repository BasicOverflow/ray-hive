"""Cluster-wide claim registry (collision avoidance + liveness GC)."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import ray

from ray_hive.cluster.project import SYSTEM_NAMESPACE, ProjectContext

REGISTRY_ACTOR = "hive_cluster_registry"


class ClaimKind(str, Enum):
    SERVE_ROUTE = "serve_route"
    MODEL_ID = "model_id"
    NAMED_ACTOR = "named_actor"
    CRON_JOB = "cron_job_id"
    TRAIN_RUN = "train_run_id"
    QUEUE = "queue"


class EntityState(str, Enum):
    ALIVE = "alive"
    SUSPECT = "suspect"
    DEAD = "dead"


@dataclass
class Claim:
    kind: str
    name: str
    project_id: str
    state: str = EntityState.ALIVE.value
    last_heartbeat_at: float = 0.0
    lease_until: float = 0.0
    meta: dict | None = None


@dataclass
class ProjectRecord:
    project_id: str
    namespace: str
    serve_route_prefix: str | None
    model_id_prefix: str | None
    scratch_prefix: str | None
    state: str = EntityState.ALIVE.value
    last_heartbeat_at: float = 0.0
    meta: dict | None = None


class ClusterRegistryLogic:
    """In-process registry state (also used by the Ray actor)."""

    def __init__(self, heartbeat_miss_s: float = 120.0, grace_s: float = 300.0):
        self._claims: dict[tuple[str, str], Claim] = {}
        self._projects: dict[str, ProjectRecord] = {}
        self._heartbeat_miss_s = heartbeat_miss_s
        self._grace_s = grace_s

    def register_project(self, ctx: dict) -> ProjectRecord:
        now = time.time()
        rec = ProjectRecord(
            project_id=ctx["project_id"],
            namespace=ctx["namespace"],
            serve_route_prefix=ctx.get("serve_route_prefix"),
            model_id_prefix=ctx.get("model_id_prefix"),
            scratch_prefix=ctx.get("scratch_prefix"),
            state=EntityState.ALIVE.value,
            last_heartbeat_at=now,
            meta=ctx.get("meta"),
        )
        self._projects[rec.project_id] = rec
        return rec

    def heartbeat_project(self, project_id: str) -> bool:
        rec = self._projects.get(project_id)
        if not rec:
            return False
        rec.last_heartbeat_at = time.time()
        rec.state = EntityState.ALIVE.value
        return True

    def claim(
        self,
        kind: str,
        name: str,
        project_id: str,
        *,
        lease_s: float = 3600.0,
        meta: dict | None = None,
    ) -> Claim:
        key = (kind, name)
        now = time.time()
        existing = self._claims.get(key)
        if existing and existing.project_id != project_id and existing.state != EntityState.DEAD.value:
            if existing.lease_until > now:
                raise RuntimeError(
                    f"claim collision kind={kind} name={name} "
                    f"held_by={existing.project_id} requester={project_id}"
                )
        c = Claim(
            kind=kind,
            name=name,
            project_id=project_id,
            state=EntityState.ALIVE.value,
            last_heartbeat_at=now,
            lease_until=now + lease_s,
            meta=meta or {},
        )
        self._claims[key] = c
        return c

    def release(self, kind: str, name: str, project_id: str) -> bool:
        key = (kind, name)
        c = self._claims.get(key)
        if not c or c.project_id != project_id:
            return False
        del self._claims[key]
        return True

    def heartbeat_claim(self, kind: str, name: str, project_id: str, *, lease_s: float = 3600.0) -> bool:
        key = (kind, name)
        c = self._claims.get(key)
        if not c or c.project_id != project_id:
            return False
        now = time.time()
        c.last_heartbeat_at = now
        c.lease_until = now + lease_s
        c.state = EntityState.ALIVE.value
        return True

    def list_claims(self, project_id: str | None = None) -> list[dict]:
        rows = list(self._claims.values())
        if project_id:
            rows = [c for c in rows if c.project_id == project_id]
        return [asdict(c) for c in rows]

    def list_projects(self) -> list[dict]:
        return [asdict(p) for p in self._projects.values()]

    def get_project(self, project_id: str) -> dict | None:
        p = self._projects.get(project_id)
        return asdict(p) if p else None

    def gc(self, now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        suspects = 0
        dead = 0
        purged = 0
        for p in list(self._projects.values()):
            age = now - p.last_heartbeat_at
            if age > self._heartbeat_miss_s + self._grace_s:
                if p.state != EntityState.DEAD.value:
                    p.state = EntityState.DEAD.value
                    dead += 1
                for key, c in list(self._claims.items()):
                    if c.project_id == p.project_id:
                        del self._claims[key]
                        purged += 1
                del self._projects[p.project_id]
            elif age > self._heartbeat_miss_s:
                if p.state != EntityState.SUSPECT.value:
                    p.state = EntityState.SUSPECT.value
                    suspects += 1
        for key, c in list(self._claims.items()):
            if c.lease_until < now - self._grace_s:
                del self._claims[key]
                purged += 1
            elif c.lease_until < now:
                c.state = EntityState.SUSPECT.value
                suspects += 1
        return {"suspects": suspects, "dead": dead, "purged": purged}


@ray.remote
class ClusterRegistry:
    def __init__(self, heartbeat_miss_s: float = 120.0, grace_s: float = 300.0):
        self._s = ClusterRegistryLogic(heartbeat_miss_s, grace_s)

    def register_project(self, ctx: dict):
        return self._s.register_project(ctx)

    def heartbeat_project(self, project_id: str):
        return self._s.heartbeat_project(project_id)

    def claim(self, kind: str, name: str, project_id: str, lease_s: float = 3600.0, meta: dict | None = None):
        return self._s.claim(kind, name, project_id, lease_s=lease_s, meta=meta)

    def release(self, kind: str, name: str, project_id: str):
        return self._s.release(kind, name, project_id)

    def heartbeat_claim(self, kind: str, name: str, project_id: str, lease_s: float = 3600.0):
        return self._s.heartbeat_claim(kind, name, project_id, lease_s=lease_s)

    def list_claims(self, project_id: str | None = None):
        return self._s.list_claims(project_id)

    def list_projects(self):
        return self._s.list_projects()

    def get_project(self, project_id: str):
        return self._s.get_project(project_id)

    def gc(self, now: float | None = None):
        return self._s.gc(now)


def get_cluster_registry(*, create: bool = True) -> Any:
    try:
        return ray.get_actor(REGISTRY_ACTOR, namespace=SYSTEM_NAMESPACE)
    except ValueError:
        if not create:
            raise
        return ClusterRegistry.options(
            name=REGISTRY_ACTOR,
            namespace=SYSTEM_NAMESPACE,
            lifetime="detached",
            max_restarts=-1,
            max_concurrency=1,
        ).remote()


def register_project(ctx: ProjectContext) -> dict:
    reg = get_cluster_registry()
    return ray.get(
        reg.register_project.remote(
            {
                "project_id": ctx.project_id,
                "namespace": ctx.namespace,
                "serve_route_prefix": ctx.serve_route_prefix,
                "model_id_prefix": ctx.model_id_prefix,
                "scratch_prefix": ctx.scratch_prefix,
                "meta": ctx.meta,
            }
        )
    )
