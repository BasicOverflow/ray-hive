"""Batch submission with ray.wait backpressure + AdmissionBroker window."""
from __future__ import annotations

import logging
from typing import Any, Callable

import ray

from ray_hive.jobs.admission import TaskShape, get_admission_broker

logger = logging.getLogger("ray_hive.jobs.fanout")


def chunks(items: list, size: int) -> list[list]:
    if size <= 0:
        size = 1
    return [items[i : i + size] for i in range(0, len(items), size)]


class BatchFanout:
    def __init__(
        self,
        *,
        use_broker: bool = True,
        on_item_error: Callable[[Exception, Any], None] | None = None,
    ):
        self.use_broker = use_broker
        self.on_item_error = on_item_error

    def run(
        self,
        label: str,
        items: list,
        submit_fn: Callable,
        shape: TaskShape,
        *,
        on_batch_done: Callable[[Any], None] | None = None,
    ) -> list:
        return run_batches(
            label,
            items,
            submit_fn,
            shape,
            on_batch_done=on_batch_done,
            use_broker=self.use_broker,
            on_item_error=self.on_item_error,
        )


def run_batches(
    label: str,
    items: list,
    submit_fn: Callable,
    shape: TaskShape,
    *,
    on_batch_done: Callable[[Any], None] | None = None,
    use_broker: bool = True,
    on_item_error: Callable[[Exception, Any], None] | None = None,
) -> list:
    batches = chunks(items, shape.batch_size)
    results: list = []
    pending: list = []
    next_i = 0
    done_batches = 0
    broker = None
    if use_broker:
        try:
            broker = get_admission_broker()
        except Exception:
            broker = None

    def _cap() -> int:
        if broker is not None:
            try:
                return int(ray.get(broker.admit.remote(shape)))
            except Exception:
                pass
        if shape.max_in_flight and shape.max_in_flight > 0:
            return shape.max_in_flight
        return max(1, min(shape.max_ceiling or 64, len(batches)))

    logger.info("%s start items=%s batches=%s", label, len(items), len(batches))
    while next_i < len(batches) or pending:
        while next_i < len(batches) and len(pending) < _cap():
            batch = batches[next_i]
            next_i += 1
            pending.append(submit_fn(batch))
            if broker is not None:
                try:
                    ray.get(broker.book.remote(cpus=shape.cpus_per_task))
                except Exception:
                    pass
        if not pending:
            break
        ready, pending = ray.wait(pending, num_returns=1)
        ref = ready[0]
        try:
            val = ray.get(ref)
            results.append(val)
            if on_batch_done:
                on_batch_done(val)
            done_batches += 1
            if broker is not None:
                try:
                    ray.get(broker.unbook.remote(cpus=shape.cpus_per_task))
                    if len(pending) == 0:
                        ray.get(broker.note_success_idle.remote())
                except Exception:
                    pass
        except Exception as exc:
            if broker is not None:
                kind = "oom" if "out of memory" in str(exc).lower() or "WorkerDied" in type(exc).__name__ else "error"
                try:
                    ray.get(broker.note_failure.remote(kind))
                    ray.get(broker.unbook.remote(cpus=shape.cpus_per_task))
                except Exception:
                    pass
            if on_item_error:
                on_item_error(exc, ref)
            else:
                raise
    logger.info("%s done batches=%s results=%s", label, done_batches, len(results))
    return results
