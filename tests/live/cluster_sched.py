"""Live-cluster GPU claim/release for parallel Deploy cycles.

Reads a fresh VRAM snapshot at every claim attempt and only grants
disjoint GPU keys across in-flight cycles in this pytest process.

Claims wait while sibling cycles still hold GPUs. Timeout only applies when
nothing in-process is holding capacity (external busy / true shortage).
Structurally impossible needs (e.g. 2 Ada hosts when the cluster has one)
fail immediately.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable


@dataclass
class GpuNeed:
    """Resource request for one live cycle."""
    min_free_gb: float
    count: int = 1
    same_host: bool = False
    distinct_hosts: bool = False
    exclusive: bool = True
    min_compute_cap: tuple[int, int] | None = None
    name: str = "cycle"


@dataclass
class Claim:
    need: GpuNeed
    gpu_keys: list[str]
    claim_id: int


class UnsatisfiableError(TimeoutError):
    """Cluster inventory can never meet this need (not a transient wait)."""


def _gpu_compute_cap(info: dict) -> tuple[int, int]:
    specs = info.get("specs") or {}
    return (
        int(specs.get("compute_capability_major", 0)),
        int(specs.get("compute_capability_minor", 0)),
    )


class ClusterScheduler:
    """Process-local GPU claims over a live (or fake) VRAM snapshot fn."""

    def __init__(
        self,
        get_vram_state: Callable[[], dict],
        *,
        poll_s: float = 2.0,
        max_wait_s: float = 90.0,
    ):
        self._get = get_vram_state
        self.poll_s = poll_s
        self.max_wait_s = max_wait_s
        self._lock = threading.Lock()
        self._held: dict[int, set[str]] = {}
        self._next_id = 1
        self._release_event = threading.Event()


    def _held_keys(self) -> set[str]:
        held = set()
        for keys in self._held.values():
            held |= keys
        return held


    def _free_keys(self, state: dict) -> dict[str, float]:
        held = self._held_keys()
        out = {}
        for key, info in state.items():
            if key in held:
                continue
            avail = float(info.get("available", info.get("free", 0.0)))
            out[key] = avail
        return out


    def structurally_possible(self, need: GpuNeed, state: dict | None = None) -> bool:
        """True if cluster hardware could meet need when all GPUs are free."""
        state = state if state is not None else self._get()
        # Pretend every GPU has its full total free (ignore live load + holds).
        fake_free = {}
        for key, info in state.items():
            total = float(info.get("total", 0.0))
            if total >= need.min_free_gb:
                fake_free[key] = total
        return self._pick(state, fake_free, need) is not None


    def try_claim(self, need: GpuNeed) -> Claim | None:
        """Attempt one claim against a fresh snapshot. None if not enough now."""
        with self._lock:
            state = self._get()
            free = self._free_keys(state)
            keys = self._pick(state, free, need)
            if keys is None:
                return None
            cid = self._next_id
            self._next_id += 1
            self._held[cid] = set(keys)
            self._release_event.clear()
            return Claim(need=need, gpu_keys=keys, claim_id=cid)


    def claim(self, need: GpuNeed) -> Claim:
        """Wait until claim succeeds.

        - Immediate UnsatisfiableError if the cluster can never meet the need.
        - While sibling cycles hold GPUs, keep waiting past max_wait_s.
        - Timeout only when max_wait_s elapses with no in-process holds.
        """
        state = self._get()
        if not self.structurally_possible(need, state):
            raise UnsatisfiableError(
                f"cluster cannot satisfy {need.name}: count={need.count} "
                f">={need.min_free_gb}GB same_host={need.same_host} "
                f"distinct_hosts={need.distinct_hosts} cap>={need.min_compute_cap}"
            )

        idle_deadline = time.time() + self.max_wait_s
        while True:
            c = self.try_claim(need)
            if c is not None:
                return c

            with self._lock:
                held = bool(self._held)

            if not held and time.time() >= idle_deadline:
                state = self._get()
                free = self._free_keys(state)
                raise TimeoutError(
                    f"no GPUs for {need.name}: need count={need.count} "
                    f">={need.min_free_gb}GB same_host={need.same_host} "
                    f"distinct_hosts={need.distinct_hosts} "
                    f"cap>={need.min_compute_cap}; "
                    f"free_now={ {k: round(v, 2) for k, v in free.items()} }"
                )

            # Sibling still holding — wait for a release (or poll).
            self._release_event.wait(timeout=self.poll_s)
            self._release_event.clear()
            if held:
                # Extend idle deadline so external timeout only counts quiet time.
                idle_deadline = time.time() + self.max_wait_s


    def release(self, claim: Claim | int) -> None:
        cid = claim.claim_id if isinstance(claim, Claim) else claim
        with self._lock:
            self._held.pop(cid, None)
        self._release_event.set()


    def _pick(self, state: dict, free: dict[str, float], need: GpuNeed) -> list[str] | None:
        eligible = []
        for k, gb in free.items():
            if k not in state:
                continue
            if gb < need.min_free_gb:
                continue
            if need.min_compute_cap is not None:
                if _gpu_compute_cap(state[k]) < need.min_compute_cap:
                    continue
            eligible.append(k)
        eligible.sort(key=lambda k: (-free[k], k))
        if len(eligible) < need.count:
            return None
        if need.same_host:
            by_host: dict[str, list[str]] = {}
            for k in eligible:
                by_host.setdefault(k.split(":")[0], []).append(k)
            for host in sorted(by_host, key=lambda h: (-len(by_host[h]), h)):
                gpus = by_host[host]
                if len(gpus) >= need.count:
                    return gpus[: need.count]
            return None
        if need.distinct_hosts:
            picked = []
            used_hosts = set()
            for k in eligible:
                host = k.split(":")[0]
                if host in used_hosts:
                    continue
                picked.append(k)
                used_hosts.add(host)
                if len(picked) == need.count:
                    return picked
            return None
        return eligible[: need.count]


def run_cycles_parallel(
    scheduler: ClusterScheduler,
    cycles: list[tuple[GpuNeed, Callable[[Claim], None]]],
    *,
    max_workers: int | None = None,
) -> list[BaseException | None]:
    """Claim+run each cycle in a thread pool; release in finally.

    Returns per-cycle exception (None on success). TimeoutError → skip-style error.
    """
    workers = max_workers or min(8, max(1, len(cycles)))
    results: list[BaseException | None] = [None] * len(cycles)

    def _wrap(idx: int, need: GpuNeed, fn: Callable[[Claim], None]):
        claim = None
        try:
            claim = scheduler.claim(need)
            fn(claim)
        except BaseException as e:
            results[idx] = e
        finally:
            if claim is not None:
                scheduler.release(claim)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [
            pool.submit(_wrap, i, need, fn)
            for i, (need, fn) in enumerate(cycles)
        ]
        for f in as_completed(futs):
            f.result()
    return results
