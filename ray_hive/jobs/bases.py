"""Thin abstract shapes for job primitives."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RemoteTaskSpec:
    """Envelope + retry hints for a remote callable."""

    name: str
    envelope: str = "cpu"  # io | cpu | ops | gpu
    max_retries: int = 2
    num_cpus: float | int = 1
    memory: int | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class NamedActorSpec:
    """Get-or-create named actor identity."""

    name: str
    namespace: str | None = None
    max_restarts: int = -1
    max_concurrency: int = 1


@dataclass
class DurableWorkItem:
    """Queue payload with idempotency key."""

    job_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
    merge_key: str | None = None
