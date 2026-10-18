"""CronBoardLogic + schedule next-fire."""
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from ray_hive.jobs.cron import CronBoardLogic, _parse_next


def test_parse_hourly():
    now = datetime(2024, 1, 1, 10, 30, tzinfo=ZoneInfo("UTC"))
    ts = _parse_next("hourly", "UTC", after=now)
    assert ts > now.timestamp()


def test_board_due_and_mark():
    board = CronBoardLogic()
    board.upsert(
        {
            "cron_id": "eod",
            "project_id": "markets",
            "schedule": "every_1_m",
            "tz": "UTC",
            "load_class": "io",
            "next_fire": time.time() - 10,
            "enabled": True,
        }
    )
    due = board.due()
    assert len(due) == 1
    board.mark_fire("eod", "ok")
    assert board.list_entries()[0]["last_status"] == "ok"


def test_remove_project():
    board = CronBoardLogic()
    board.upsert(
        {"cron_id": "a", "project_id": "p1", "schedule": "hourly", "enabled": True}
    )
    board.upsert(
        {"cron_id": "b", "project_id": "p2", "schedule": "hourly", "enabled": True}
    )
    assert board.remove_project("p1") == 1
    assert len(board.list_entries()) == 1
