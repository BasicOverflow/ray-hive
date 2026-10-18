"""Durable done-set for long multi-item jobs."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class JobProgress:
    def __init__(self, db_path: str | Path, job_key: str):
        self.db_path = str(Path(db_path))
        self.job_key = job_key
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS done ("
                "job_key TEXT NOT NULL, item_id TEXT NOT NULL, meta_json TEXT, "
                "PRIMARY KEY (job_key, item_id))"
            )

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def mark_done(self, item_id: str, meta: dict | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO done(job_key, item_id, meta_json) VALUES (?,?,?)",
                (self.job_key, str(item_id), json.dumps(meta or {})),
            )

    def is_done(self, item_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM done WHERE job_key=? AND item_id=?",
                (self.job_key, str(item_id)),
            ).fetchone()
        return row is not None

    def done_set(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT item_id FROM done WHERE job_key=?", (self.job_key,)
            ).fetchall()
        return {r[0] for r in rows}

    def pending(self, items: list[str]) -> list[str]:
        done = self.done_set()
        return [i for i in items if str(i) not in done]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM done WHERE job_key=?", (self.job_key,))
