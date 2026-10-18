"""Generational per-worker scratch store with end flush hook."""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Callable

import ray

from ray_hive.jobs.envelopes import io_options


@ray.remote(**io_options())
class ScratchShardActor:
    def __init__(self, root: str, generation: str, shard_id: str):
        self.root = Path(root) / generation / shard_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.generation = generation
        self.shard_id = shard_id

    def path(self) -> str:
        return str(self.root)

    def write_bytes(self, name: str, data: bytes) -> str:
        p = self.root / name
        p.write_bytes(data)
        return str(p)

    def list_files(self) -> list[str]:
        return [str(p) for p in self.root.rglob("*") if p.is_file()]

    def wipe(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)


class ScratchShardStore:
    """Detached local scratch shards keyed by generation."""

    def __init__(
        self,
        project_id: str,
        *,
        root: str | None = None,
        generation: str | None = None,
        namespace: str | None = None,
    ):
        self.project_id = project_id
        self.root = root or f"/tmp/{project_id}-scratch"
        self.generation = generation or uuid.uuid4().hex[:12]
        self.namespace = namespace
        self._actors: dict[str, object] = {}

    def shard(self, shard_id: str):
        if shard_id in self._actors:
            return self._actors[shard_id]
        name = f"{self.project_id}-scratch-{self.generation}-{shard_id}"
        opts = dict(name=name, lifetime="detached", **io_options())
        if self.namespace:
            opts["namespace"] = self.namespace
        try:
            actor = ray.get_actor(name, namespace=self.namespace) if self.namespace else ray.get_actor(name)
        except ValueError:
            actor = ScratchShardActor.options(**opts).remote(
                self.root, self.generation, shard_id
            )
        self._actors[shard_id] = actor
        return actor

    def flush_all(self, flush_fn: Callable[[str, list[str]], None]) -> None:
        for shard_id, actor in self._actors.items():
            files = ray.get(actor.list_files.remote())
            flush_fn(shard_id, files)

    def abandon(self) -> None:
        for actor in self._actors.values():
            try:
                ray.get(actor.wipe.remote())
            except Exception:
                pass
        self._actors.clear()
