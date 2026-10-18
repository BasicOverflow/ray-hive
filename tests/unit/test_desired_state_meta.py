"""Meta-supervisor DesiredState tables."""
import time

from ray_hive.supervisor.desired_state import (
    DesiredState,
    StickyCron,
    StickyProject,
)


def test_project_gc(tmp_path):
    state = DesiredState(tmp_path / "s.db")
    state.upsert_project(
        StickyProject(
            project_id="p1",
            namespace="ns",
            last_heartbeat_at=0.0,
        )
    )
    state.upsert_cron(
        StickyCron(cron_id="c1", project_id="p1", schedule="hourly")
    )
    dead = state.gc_stale_projects(miss_s=1, grace_s=1, now=100.0)
    assert dead == ["p1"]
    assert state.list_projects() == []
    assert state.list_crons() == []


def test_cron_persist(tmp_path):
    state = DesiredState(tmp_path / "s.db")
    state.upsert_cron(
        StickyCron(
            cron_id="eod",
            project_id="markets",
            schedule="16:05",
            next_fire=time.time() + 100,
        )
    )
    assert state.list_crons()[0].schedule == "16:05"
