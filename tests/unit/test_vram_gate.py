"""VRAM wait helper (fake snapshot)."""
from unittest.mock import patch

import pytest

from tests.helpers import make_gpu
from tests.live.vram_gate import wait_for_available


def test_wait_for_available_succeeds():
    state = {"h:gpu0": make_gpu("h:gpu0", 0.0, 24)}
    n = {"i": 0}

    def get():
        n["i"] += 1
        if n["i"] >= 2:
            state["h:gpu0"] = make_gpu("h:gpu0", 10.0, 24)
        return state

    with patch("tests.live.vram_gate.time.sleep"):
        assert wait_for_available(get, "h:gpu0", 4.0, poll_s=0.01, max_wait_s=1.0) >= 4.0


def test_wait_for_available_timeout():
    state = {"h:gpu0": make_gpu("h:gpu0", 0.0, 24)}
    with patch("tests.live.vram_gate.time.sleep"), pytest.raises(TimeoutError):
        wait_for_available(lambda: state, "h:gpu0", 4.0, poll_s=0.01, max_wait_s=0.05)
