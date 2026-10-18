"""Chaos persist — power-cycle Ray VMs. Opt-in only."""
from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.chaos, pytest.mark.live]


def test_chaos_persist_runbook_gate():
    """Skipped unless HIVE_CHAOS=1. See scripts/chaos_ray_vms.md."""
    if os.environ.get("HIVE_CHAOS") != "1":
        pytest.skip("set HIVE_CHAOS=1 and follow scripts/chaos_ray_vms.md")
    # Homelab automation hooks go here (qm stop/start via plink).
    # Intentionally not auto-running destructive VM power cycles in default CI.
    assert os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "chaos_ray_vms.md")
    )
