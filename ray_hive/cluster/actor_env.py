"""Runtime env for hive system actors created via Ray Client.

Head/CPU/GPU worker images should bake ``ray-hive`` in. This env is the
fallback so meta/project supervisors can still create ClusterRegistry /
AdmissionBroker / CronBoard while a rolling image update is in flight.
Gitea Packages simple index is LAN-public for downloads.
"""
from __future__ import annotations

HIVE_PIP_EXTRA_INDEX = "http://10.0.0.52:3000/api/packages/admin/pypi/simple/"
HIVE_PACKAGE = "ray-hive==0.2.1"


def system_actor_runtime_env() -> dict:
    return {
        "pip": [HIVE_PACKAGE],
        "env_vars": {
            "PIP_EXTRA_INDEX_URL": HIVE_PIP_EXTRA_INDEX,
            "PIP_INDEX_URL": "https://pypi.org/simple",
        },
    }
