"""DurableJobQueue + JobProgress."""
from ray_hive.jobs.progress import JobProgress
from ray_hive.jobs.queue import DurableJobQueue


def test_queue_enqueue_claim_complete(tmp_path):
    q = DurableJobQueue(tmp_path / "q.db", stale_running_s=3600)
    i = q.enqueue("eod", {"day": "2024-01-01"})
    row = q.claim_next(["eod"])
    assert row["id"] == i
    assert row["status"] == "running"
    q.complete(i)
    assert q.claim_next() is None


def test_queue_merge_key(tmp_path):
    q = DurableJobQueue(tmp_path / "q.db")
    a = q.enqueue("compact", {"months": [1]}, merge_key="compact")
    b = q.enqueue("compact", {"months": [2]}, merge_key="compact")
    assert a == b
    row = q.claim_next()
    assert set(row["payload"]["months"]) == {1, 2}


def test_queue_stale_requeue(tmp_path):
    q = DurableJobQueue(tmp_path / "q.db", stale_running_s=1)
    i = q.enqueue("seed", {})
    q.claim_next()
    # force old started_at
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "q.db"))
    conn.execute("UPDATE pending SET started_at='2000-01-01T00:00:00+00:00' WHERE id=?", (i,))
    conn.commit()
    conn.close()
    n = q.requeue_stale()
    assert n == 1
    row = q.claim_next()
    assert row["id"] == i


def test_progress_pending(tmp_path):
    p = JobProgress(tmp_path / "p.db", "embed")
    p.mark_done("a")
    assert p.pending(["a", "b", "c"]) == ["b", "c"]
