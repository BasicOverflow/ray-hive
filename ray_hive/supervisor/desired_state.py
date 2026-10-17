"""SQLite sticky desired-state for hive models (survives Ray reconnects)."""
from __future__ import annotations

import json
import os
import sqlite3
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


class DesiredState:
    def __init__(self, path: str | Path | None = None):
        env = os.environ.get("HIVE_SUPERVISOR_STATE")
        self.path = Path(path or env or ".hive-supervisor/state.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS sticky ("
            "model_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
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
            out.append(StickyModel(**d))
        return out
