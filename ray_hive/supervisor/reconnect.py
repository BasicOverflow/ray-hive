"""Reconnect / Serve cool-down policy for the hive supervisor (pure, unit-tested)."""
from __future__ import annotations

from dataclasses import dataclass, field


def is_thread_exhaustion(exc: BaseException | str) -> bool:
    msg = str(exc).lower()
    return "can't start new thread" in msg or "cannot start new thread" in msg


@dataclass
class ReconnectPolicy:
    """Backoff Ray reconnects and cool down destructive Serve redeploys."""

    base_s: float = 30.0
    max_s: float = 300.0
    serve_cooldown_s: float = 600.0
    thread_fail_exit_after: int = 2

    failures: int = 0
    next_attempt_mono: float = 0.0
    last_serve_attempt_mono: float = field(default=-1e18)
    thread_failures: int = 0

    def note_success(self) -> None:
        self.failures = 0
        self.thread_failures = 0
        self.next_attempt_mono = 0.0

    def note_failure(self, now: float, *, thread_exhausted: bool = False) -> float:
        self.failures += 1
        delay = min(self.max_s, self.base_s * (2 ** max(0, self.failures - 1)))
        self.next_attempt_mono = now + delay
        if thread_exhausted:
            self.thread_failures += 1
        else:
            self.thread_failures = 0
        return delay

    def should_attempt(self, now: float) -> bool:
        return now >= self.next_attempt_mono

    def should_exit_wedged(self) -> bool:
        return self.thread_failures >= self.thread_fail_exit_after

    def should_attempt_serve(self, now: float) -> bool:
        if self.last_serve_attempt_mono < 0:
            return True
        return (now - self.last_serve_attempt_mono) >= self.serve_cooldown_s

    def note_serve_attempt(self, now: float) -> None:
        self.last_serve_attempt_mono = now
