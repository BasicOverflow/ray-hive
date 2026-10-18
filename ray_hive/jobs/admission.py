"""Live in-flight admission from Ray resources (no Prometheus)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import ray

from ray_hive.cluster.project import SYSTEM_NAMESPACE

BROKER_ACTOR = "hive_admission_broker"


@dataclass
class TaskShape:
    batch_size: int = 50
    max_in_flight: int = 0  # 0 = broker-driven; >0 is a ceiling
    max_ceiling: int = 256
    mem_bytes_hint: int = 0
    cpus_per_task: float = 1.0


@ray.remote(num_cpus=0.01)
def _probe_free_ram() -> dict:
    """Cheap worker probe — free RAM / RSS via /proc when available."""
    import os

    out = {"pid": os.getpid(), "free_ram": None, "rss": None}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    out["free_ram"] = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["rss"] = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    return out


class AdmissionBrokerLogic:
    def __init__(self):
        self._window = 8
        self._min_window = 1
        self._booked_cpus = 0.0
        self._booked_mem = 0
        self._last_probe = 0.0
        self._free_ram_hint: int | None = None
        self._shrink_factor = 0.5
        self._grow_step = 2

    def _cluster_cpus(self) -> float:
        try:
            avail = ray.available_resources()
            return float(avail.get("CPU", 0.0))
        except Exception:
            return 0.0

    def _maybe_probe(self) -> None:
        now = time.time()
        if now - self._last_probe < 15.0:
            return
        self._last_probe = now
        try:
            ref = _probe_free_ram.options(
                scheduling_strategy="SPREAD",
                label_selector={"ray.io/node-group": "!headgroup"},
            ).remote()
            info = ray.get(ref, timeout=10)
            if info.get("free_ram"):
                self._free_ram_hint = int(info["free_ram"])
        except Exception:
            pass

    def suggest_cpus(self, default: float = 1.0) -> float:
        free = self._cluster_cpus()
        if free >= 4:
            return default
        if free >= 1:
            return min(default, 1.0)
        return 0.25

    def admit(self, shape: TaskShape | dict) -> int:
        if isinstance(shape, dict):
            shape = TaskShape(**{k: v for k, v in shape.items() if k in TaskShape.__dataclass_fields__})
        self._maybe_probe()
        free_cpu = max(0.0, self._cluster_cpus() - self._booked_cpus)
        by_cpu = int(free_cpu // max(shape.cpus_per_task, 0.001)) if shape.cpus_per_task else self._window
        n = max(self._min_window, min(self._window, by_cpu if by_cpu > 0 else self._window))
        if shape.max_ceiling > 0:
            n = min(n, shape.max_ceiling)
        if shape.max_in_flight and shape.max_in_flight > 0:
            n = min(n, shape.max_in_flight)
        if self._free_ram_hint is not None and shape.mem_bytes_hint > 0:
            by_mem = max(1, int(self._free_ram_hint * 0.5 // shape.mem_bytes_hint))
            n = min(n, by_mem)
        return max(1, n)

    def book(self, cpus: float = 0.0, mem: int = 0) -> None:
        self._booked_cpus += cpus
        self._booked_mem += mem

    def unbook(self, cpus: float = 0.0, mem: int = 0) -> None:
        self._booked_cpus = max(0.0, self._booked_cpus - cpus)
        self._booked_mem = max(0, self._booked_mem - mem)

    def note_failure(self, kind: str = "oom") -> int:
        self._window = max(self._min_window, int(self._window * self._shrink_factor))
        return self._window

    def note_success_idle(self) -> int:
        free = self._cluster_cpus()
        if free > 2:
            self._window = min(256, self._window + self._grow_step)
        return self._window

    def snapshot(self) -> dict[str, Any]:
        return {
            "window": self._window,
            "booked_cpus": self._booked_cpus,
            "booked_mem": self._booked_mem,
            "free_ram_hint": self._free_ram_hint,
            "cluster_cpus": self._cluster_cpus(),
        }


@ray.remote
class AdmissionBroker:
    def __init__(self):
        self._s = AdmissionBrokerLogic()

    def suggest_cpus(self, default: float = 1.0):
        return self._s.suggest_cpus(default)

    def admit(self, shape):
        return self._s.admit(shape)

    def book(self, cpus: float = 0.0, mem: int = 0):
        return self._s.book(cpus, mem)

    def unbook(self, cpus: float = 0.0, mem: int = 0):
        return self._s.unbook(cpus, mem)

    def note_failure(self, kind: str = "oom"):
        return self._s.note_failure(kind)

    def note_success_idle(self):
        return self._s.note_success_idle()

    def snapshot(self):
        return self._s.snapshot()


def get_admission_broker(*, create: bool = True):
    try:
        return ray.get_actor(BROKER_ACTOR, namespace=SYSTEM_NAMESPACE)
    except ValueError:
        if not create:
            raise
        return AdmissionBroker.options(
            name=BROKER_ACTOR,
            namespace=SYSTEM_NAMESPACE,
            lifetime="detached",
            max_restarts=-1,
            max_concurrency=1,
        ).remote()
