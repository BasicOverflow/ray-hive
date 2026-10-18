"""Offline-safe SQLite job queue."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS pending (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    merge_key TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_pending_status ON pending (status, created_at);
CREATE INDEX IF NOT EXISTS idx_pending_merge ON pending (merge_key, status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DurableJobQueue:
    def __init__(self, db_path: str | Path, *, stale_running_s: float = 4 * 3600):
        self.db_path = str(Path(db_path))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.stale_running_s = stale_running_s
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def enqueue(
        self,
        job_type: str,
        payload: dict | None = None,
        *,
        merge_key: str | None = None,
    ) -> int:
        if merge_key:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT id, payload_json FROM pending "
                    "WHERE merge_key=? AND status='pending'",
                    (merge_key,),
                ).fetchone()
                if row:
                    existing = json.loads(row["payload_json"] or "{}")
                    merged = {**existing, **(payload or {})}
                    # list-merge for months-like keys
                    for k, v in (payload or {}).items():
                        if isinstance(v, list) and isinstance(existing.get(k), list):
                            merged[k] = existing[k] + [x for x in v if x not in existing[k]]
                    conn.execute(
                        "UPDATE pending SET payload_json=? WHERE id=?",
                        (json.dumps(merged), row["id"]),
                    )
                    return int(row["id"])
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO pending (job_type, payload_json, status, merge_key, created_at) "
                "VALUES (?, ?, 'pending', ?, ?)",
                (job_type, json.dumps(payload or {}), merge_key, _now()),
            )
            return int(cur.lastrowid)

    def claim_next(self, job_types: list[str] | None = None) -> dict | None:
        self.requeue_stale()
        with self._connect() as conn:
            if job_types:
                q = ",".join("?" * len(job_types))
                row = conn.execute(
                    f"SELECT * FROM pending WHERE status='pending' AND job_type IN ({q}) "
                    "ORDER BY id ASC LIMIT 1",
                    tuple(job_types),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM pending WHERE status='pending' ORDER BY id ASC LIMIT 1"
                ).fetchone()
            if not row:
                return None
            conn.execute(
                "UPDATE pending SET status='running', started_at=? WHERE id=?",
                (_now(), row["id"]),
            )
            return self._row(row, status="running")

    def complete(self, job_id: int, *, error: str | None = None) -> None:
        status = "failed" if error else "done"
        with self._connect() as conn:
            conn.execute(
                "UPDATE pending SET status=?, finished_at=?, error=? WHERE id=?",
                (status, _now(), error, job_id),
            )

    def requeue_stale(self, now: float | None = None) -> int:
        now = now if now is not None else time.time()
        cutoff = datetime.fromtimestamp(now - self.stale_running_s, tz=timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE pending SET status='pending', started_at=NULL, error='stale_requeue' "
                "WHERE status='running' AND started_at IS NOT NULL AND started_at < ?",
                (cutoff,),
            )
            return cur.rowcount or 0

    def list_pending(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pending WHERE status IN ('pending','running') ORDER BY id"
            ).fetchall()
        return [self._row(r) for r in rows]

    def counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM pending GROUP BY status"
            ).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    @staticmethod
    def _row(row: sqlite3.Row, status: str | None = None) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "job_type": row["job_type"],
            "payload": json.loads(row["payload_json"] or "{}"),
            "status": status or row["status"],
            "merge_key": row["merge_key"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"] if "finished_at" in row.keys() else None,
            "error": row["error"],
        }
