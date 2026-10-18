"""SQLite sticky desired-state for meta-supervisor (models, projects, crons, queues)."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class StickyModel:
    model_id: str
    model_name: str
    max_input_prompt_length: int | str
    max_output_prompt_length: int
    replicas: int = 1
    gpu: str | list[str] | None = None
    idle_timeout: int = -1
    sleep_timeout: int = -1
    vllm_kwargs: dict | None = None
    project_id: str | None = None


@dataclass
class StickyProject:
    project_id: str
    namespace: str
    serve_route_prefix: str | None = None
    model_id_prefix: str | None = None
    scratch_prefix: str | None = None
    last_heartbeat_at: float = 0.0
    state: str = "alive"


@dataclass
class StickyCron:
    cron_id: str
    project_id: str
    schedule: str
    tz: str = "America/New_York"
    load_class: str = "cpu"
    enabled: bool = True
    next_fire: float | None = None
    last_fire: float | None = None
    last_status: str | None = None


@dataclass
class StickyQueue:
    project_id: str
    path: str
    last_heartbeat_at: float = 0.0


@dataclass
class StickyTrain:
    run_id: str
    project_id: str
    payload: dict | None = None
    state: str = "desired"


class DesiredState:
    def __init__(self, path: str | Path | None = None):
        env = os.environ.get("HIVE_SUPERVISOR_STATE")
        self.path = Path(path or env or ".hive-supervisor/state.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sticky (
                model_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cron_board (
                cron_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS queues (
                project_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS trains (
                run_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS claims (
                kind TEXT NOT NULL, name TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY (kind, name));
            """
        )
        self._conn.commit()

    def pin(self, model: StickyModel) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sticky(model_id, payload) VALUES (?, ?)",
            (model.model_id, json.dumps(asdict(model))),
        )
        self._conn.commit()

    def unpin(self, model_id: str) -> None:
        self._conn.execute("DELETE FROM sticky WHERE model_id=?", (model_id,))
        self._conn.commit()

    def list(self) -> list[StickyModel]:
        rows = self._conn.execute("SELECT payload FROM sticky").fetchall()
        out = []
        for (payload,) in rows:
            d = json.loads(payload)
            out.append(StickyModel(**{k: v for k, v in d.items() if k in StickyModel.__dataclass_fields__}))
        return out

    def upsert_project(self, project: StickyProject) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO projects(project_id, payload) VALUES (?, ?)",
            (project.project_id, json.dumps(asdict(project))),
        )
        self._conn.commit()

    def list_projects(self) -> list[StickyProject]:
        rows = self._conn.execute("SELECT payload FROM projects").fetchall()
        return [StickyProject(**json.loads(p)) for (p,) in rows]

    def delete_project(self, project_id: str) -> None:
        self._conn.execute("DELETE FROM projects WHERE project_id=?", (project_id,))
        self._conn.execute("DELETE FROM queues WHERE project_id=?", (project_id,))
        for cron in self.list_crons():
            if cron.project_id == project_id:
                self._conn.execute("DELETE FROM cron_board WHERE cron_id=?", (cron.cron_id,))
        self._conn.commit()

    def upsert_cron(self, cron: StickyCron) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cron_board(cron_id, payload) VALUES (?, ?)",
            (cron.cron_id, json.dumps(asdict(cron))),
        )
        self._conn.commit()

    def list_crons(self) -> list[StickyCron]:
        rows = self._conn.execute("SELECT payload FROM cron_board").fetchall()
        return [StickyCron(**json.loads(p)) for (p,) in rows]

    def delete_cron(self, cron_id: str) -> None:
        self._conn.execute("DELETE FROM cron_board WHERE cron_id=?", (cron_id,))
        self._conn.commit()

    def upsert_queue(self, queue: StickyQueue) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO queues(project_id, payload) VALUES (?, ?)",
            (queue.project_id, json.dumps(asdict(queue))),
        )
        self._conn.commit()

    def list_queues(self) -> list[StickyQueue]:
        rows = self._conn.execute("SELECT payload FROM queues").fetchall()
        return [StickyQueue(**json.loads(p)) for (p,) in rows]

    def upsert_train(self, train: StickyTrain) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO trains(run_id, payload) VALUES (?, ?)",
            (train.run_id, json.dumps(asdict(train))),
        )
        self._conn.commit()

    def list_trains(self) -> list[StickyTrain]:
        rows = self._conn.execute("SELECT payload FROM trains").fetchall()
        return [StickyTrain(**json.loads(p)) for (p,) in rows]

    def upsert_claim(self, kind: str, name: str, payload: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO claims(kind, name, payload) VALUES (?, ?, ?)",
            (kind, name, json.dumps(payload)),
        )
        self._conn.commit()

    def list_claims(self) -> list[dict]:
        rows = self._conn.execute("SELECT kind, name, payload FROM claims").fetchall()
        out = []
        for kind, name, payload in rows:
            d = json.loads(payload)
            d.setdefault("kind", kind)
            d.setdefault("name", name)
            out.append(d)
        return out

    def gc_stale_projects(self, miss_s: float = 120.0, grace_s: float = 300.0, now: float | None = None) -> list[str]:
        now = now if now is not None else time.time()
        dead = []
        for p in self.list_projects():
            if now - p.last_heartbeat_at > miss_s + grace_s:
                dead.append(p.project_id)
                self.delete_project(p.project_id)
            elif now - p.last_heartbeat_at > miss_s:
                p.state = "suspect"
                self.upsert_project(p)
        return dead
