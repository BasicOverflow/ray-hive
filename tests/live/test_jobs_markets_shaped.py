"""Markets-shaped live tests — real YF/FRED slices (bounded)."""
from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.markets_shaped,
]


def _ray_addr() -> str:
    return os.environ.get("RAY_ADDRESS") or "ray://10.0.1.52:10001"


@pytest.fixture(scope="module")
def hive_jobs_session():
    from ray_hive.cluster import ProjectContext, init_ray_for_project

    ctx = ProjectContext(
        project_id="hive-live-test",
        namespace="hive-live-test",
        serve_route_prefix=None,
    )
    init_ray_for_project(_ray_addr(), ctx, claim_route=False)
    yield ctx


def test_fanout_io_on_cluster(hive_jobs_session):
    from ray_hive.jobs import BatchFanout, TaskShape, io_remote

    @io_remote()
    def _double(batch: list) -> list:
        return [x * 2 for x in batch]

    fan = BatchFanout()
    out = fan.run(
        "double",
        list(range(20)),
        lambda b: _double.remote(b),
        TaskShape(batch_size=5, max_ceiling=4, cpus_per_task=0.001),
    )
    flat = [x for batch in out for x in batch]
    assert sorted(flat) == [x * 2 for x in range(20)]


def test_yf_gated_slice(hive_jobs_session):
    yfinance = pytest.importorskip("yfinance")
    from ray_hive.jobs import get_or_create_rate_gate
    import ray

    gate = get_or_create_rate_gate(
        "hive-live-yf",
        namespace="hive-live-test",
        min_interval=0.5,
    )
    ray.get(gate.acquire.remote())
    t = yfinance.Ticker("AAPL")
    hist = t.history(period="5d")
    ray.get(gate.record_success.remote())
    assert not hist.empty


def test_cron_board_visible(hive_jobs_session):
    from ray_hive.jobs import CronSchedule, get_cron_board
    import ray

    CronSchedule(
        "hive-live-cron",
        "hive-live-test",
        "hourly",
        load_class="io",
    )
    board = get_cron_board()
    entries = ray.get(board.list_entries.remote())
    assert any(e["cron_id"] == "hive-live-cron" for e in entries)
