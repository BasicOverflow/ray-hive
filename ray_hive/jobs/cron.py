"""Timezone cron + central meta-supervisor cron board."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

import ray

from ray_hive.cluster.project import SYSTEM_NAMESPACE
from ray_hive.cluster.registry import ClaimKind, get_cluster_registry

CRON_BOARD_ACTOR = "hive_cron_board"


@dataclass
class CronEntry:
    cron_id: str
    project_id: str
    schedule: str  # "HH:MM" daily or "hourly" / "asap" / cron-like hour mark
    tz: str = "America/New_York"
    load_class: str = "cpu"  # io | cpu | gpu
    enabled: bool = True
    next_fire: float | None = None
    last_fire: float | None = None
    last_status: str | None = None
    lagged: bool = False


def _parse_next(schedule: str, tz_name: str, after: datetime | None = None) -> float:
    """Minimal schedule parser: ``HH:MM`` daily, ``hourly``, ``every_N_m``."""
    tz = ZoneInfo(tz_name)
    now = after or datetime.now(tz)
    if schedule == "hourly":
        nxt = now.replace(minute=0, second=0, microsecond=0)
        if nxt <= now:
            from datetime import timedelta

            nxt = nxt + timedelta(hours=1)
        return nxt.astimezone(timezone.utc).timestamp()
    if schedule.startswith("every_") and schedule.endswith("_m"):
        mins = int(schedule[len("every_") : -len("_m")])
        return now.astimezone(timezone.utc).timestamp() + mins * 60
    # HH:MM daily
    hh, mm = schedule.split(":")
    nxt = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
    if nxt <= now:
        from datetime import timedelta

        nxt = nxt + timedelta(days=1)
    return nxt.astimezone(timezone.utc).timestamp()


class CronBoardLogic:
    """Central schedule board state (also used by the Ray actor)."""

    def __init__(self):
        self._entries: dict[str, CronEntry] = {}

    def upsert(self, entry: dict) -> dict:
        e = CronEntry(**{k: v for k, v in entry.items() if k in CronEntry.__dataclass_fields__})
        if e.next_fire is None and e.enabled:
            e.next_fire = _parse_next(e.schedule, e.tz)
        self._entries[e.cron_id] = e
        return asdict(e)

    def remove(self, cron_id: str) -> bool:
        return self._entries.pop(cron_id, None) is not None

    def remove_project(self, project_id: str) -> int:
        keys = [k for k, e in self._entries.items() if e.project_id == project_id]
        for k in keys:
            del self._entries[k]
        return len(keys)

    def mark_fire(self, cron_id: str, status: str = "ok") -> dict | None:
        e = self._entries.get(cron_id)
        if not e:
            return None
        now = time.time()
        e.last_fire = now
        e.last_status = status
        e.lagged = False
        e.next_fire = _parse_next(e.schedule, e.tz)
        return asdict(e)

    def list_entries(self, project_id: str | None = None) -> list[dict]:
        rows = list(self._entries.values())
        if project_id:
            rows = [e for e in rows if e.project_id == project_id]
        now = time.time()
        out = []
        for e in rows:
            if e.enabled and e.next_fire and e.next_fire < now - 60:
                e.lagged = True
            out.append(asdict(e))
        return sorted(out, key=lambda d: (d.get("next_fire") is None, d.get("next_fire") or 0))

    def due(self, now: float | None = None) -> list[dict]:
        now = now if now is not None else time.time()
        return [
            asdict(e)
            for e in self._entries.values()
            if e.enabled and e.next_fire is not None and e.next_fire <= now
        ]


@ray.remote
class CronBoard:
    def __init__(self):
        self._s = CronBoardLogic()

    def upsert(self, entry: dict):
        return self._s.upsert(entry)

    def remove(self, cron_id: str):
        return self._s.remove(cron_id)

    def remove_project(self, project_id: str):
        return self._s.remove_project(project_id)

    def mark_fire(self, cron_id: str, status: str = "ok"):
        return self._s.mark_fire(cron_id, status)

    def list_entries(self, project_id: str | None = None):
        return self._s.list_entries(project_id)

    def due(self, now: float | None = None):
        return self._s.due(now)


def get_cron_board(*, create: bool = True):
    try:
        return ray.get_actor(CRON_BOARD_ACTOR, namespace=SYSTEM_NAMESPACE)
    except ValueError:
        if not create:
            raise
        from ray_hive.cluster.actor_env import system_actor_runtime_env

        return CronBoard.options(
            name=CRON_BOARD_ACTOR,
            namespace=SYSTEM_NAMESPACE,
            lifetime="detached",
            max_restarts=-1,
            max_concurrency=1,
            num_cpus=0.01,
            runtime_env=system_actor_runtime_env(),
        ).remote()


class CronSchedule:
    """Declare recurring work; registers on the meta cron board."""

    def __init__(
        self,
        cron_id: str,
        project_id: str,
        schedule: str,
        *,
        tz: str = "America/New_York",
        load_class: str = "cpu",
        on_fire: Callable[[], Any] | None = None,
        register_board: bool = True,
    ):
        self.cron_id = cron_id
        self.project_id = project_id
        self.schedule = schedule
        self.tz = tz
        self.load_class = load_class
        self.on_fire = on_fire
        self._register_board = register_board
        if register_board:
            self._register()

    def _register(self) -> None:
        board = get_cron_board()
        ray.get(
            board.upsert.remote(
                {
                    "cron_id": self.cron_id,
                    "project_id": self.project_id,
                    "schedule": self.schedule,
                    "tz": self.tz,
                    "load_class": self.load_class,
                    "enabled": True,
                }
            )
        )
        try:
            reg = get_cluster_registry()
            ray.get(
                reg.claim.remote(
                    ClaimKind.CRON_JOB.value,
                    self.cron_id,
                    self.project_id,
                )
            )
        except Exception:
            pass

    def unregister(self) -> None:
        board = get_cron_board()
        ray.get(board.remove.remote(self.cron_id))
        try:
            reg = get_cluster_registry()
            ray.get(reg.release.remote(ClaimKind.CRON_JOB.value, self.cron_id, self.project_id))
        except Exception:
            pass

    def poll_due(self) -> bool:
        board = get_cron_board()
        due = ray.get(board.due.remote())
        mine = [d for d in due if d["cron_id"] == self.cron_id]
        if not mine:
            return False
        if self.on_fire:
            try:
                self.on_fire()
                ray.get(board.mark_fire.remote(self.cron_id, "ok"))
            except Exception as exc:
                ray.get(board.mark_fire.remote(self.cron_id, f"error:{exc}"))
                raise
        else:
            ray.get(board.mark_fire.remote(self.cron_id, "ok"))
        return True
