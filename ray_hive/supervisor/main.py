"""Hive supervisor loop — LXC/systemd entrypoint.

Keeps running when Ray is down; reconnects with backoff; heals actors + sticky
models with Serve cool-down (Markets pattern).
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time

from ray_hive.core.ray_utils.session import init_ray, suppress_ray_warnings
from ray_hive.supervisor.desired_state import DesiredState
from ray_hive.supervisor.heal import heal_once
from ray_hive.supervisor.reconnect import ReconnectPolicy, is_thread_exhaustion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ray_hive.supervisor")

POLL_S = float(os.environ.get("HIVE_SUPERVISOR_POLL_S", "30"))
_running = True


def _stop(signum, _frame):
    global _running
    logger.info("stop signal=%s", signum)
    _running = False
    try:
        import ray

        ray.shutdown()
    except Exception:
        pass


def _ray_address() -> str:
    addr = os.environ.get("RAY_ADDRESS") or os.environ.get("HIVE_RAY_ADDRESS")
    if not addr:
        raise SystemExit("RAY_ADDRESS or HIVE_RAY_ADDRESS required")
    return addr


def main() -> int:
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    suppress_ray_warnings(True)
    state = DesiredState()
    policy = ReconnectPolicy()
    logger.info("hive supervisor starting state=%s", state.path)

    while _running:
        now = time.monotonic()
        if not policy.should_attempt(now):
            time.sleep(min(POLL_S, max(1.0, policy.next_attempt_mono - now)))
            continue
        try:
            init_ray(_ray_address(), suppress_logging=True)
            attempt_serve = policy.should_attempt_serve(now)
            if attempt_serve:
                policy.note_serve_attempt(now)
            heal_once(state, attempt_serve=attempt_serve)
            policy.note_success()
            logger.info("heal ok")
        except Exception as exc:
            delay = policy.note_failure(
                time.monotonic(), thread_exhausted=is_thread_exhaustion(exc)
            )
            logger.warning("heal failed (%s); retry in %.0fs", exc, delay)
            try:
                import ray

                ray.shutdown()
            except Exception:
                pass
            if policy.should_exit_wedged():
                logger.error("thread exhaustion wedge — exiting for systemd restart")
                return 1
        time.sleep(POLL_S)
    return 0


if __name__ == "__main__":
    sys.exit(main())
