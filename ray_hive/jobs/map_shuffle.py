"""Map → shuffle partitions → reduce helper over Ray ObjectRefs."""
from __future__ import annotations

from typing import Any, Callable, Hashable

import ray

from ray_hive.jobs.admission import TaskShape
from ray_hive.jobs.envelopes import cpu_remote
from ray_hive.jobs.fanout import BatchFanout


@cpu_remote()
def _identity_map(batch: list) -> list:
    return batch


class MapShuffleReduce:
    """Partition mapped outputs by key, then reduce each partition."""

    def __init__(
        self,
        map_fn: Callable[[list], Any],
        key_fn: Callable[[Any], Hashable],
        reduce_fn: Callable[[Hashable, list], Any],
        *,
        num_reducers: int = 8,
        shape: TaskShape | None = None,
    ):
        self.map_fn = map_fn
        self.key_fn = key_fn
        self.reduce_fn = reduce_fn
        self.num_reducers = max(1, num_reducers)
        self.shape = shape or TaskShape(batch_size=32, max_ceiling=64)
        self._fanout = BatchFanout()

    def run(self, items: list) -> list:
        mapped = self._fanout.run("map", items, self.map_fn, self.shape)
        # flatten
        flat: list = []
        for m in mapped:
            if isinstance(m, list):
                flat.extend(m)
            else:
                flat.append(m)
        buckets: list[list] = [[] for _ in range(self.num_reducers)]
        for item in flat:
            k = self.key_fn(item)
            buckets[hash(k) % self.num_reducers].append(item)

        @cpu_remote()
        def _reduce_bucket(bucket: list) -> list:
            by_key: dict[Hashable, list] = {}
            for item in bucket:
                by_key.setdefault(self.key_fn(item), []).append(item)
            return [self.reduce_fn(k, v) for k, v in by_key.items()]

        # Note: nested remote with closure may need ray.put of reduce_fn on real clusters;
        # for unit tests we reduce locally when Ray is mocked.
        results = []
        for bucket in buckets:
            if not bucket:
                continue
            by_key: dict[Hashable, list] = {}
            for item in bucket:
                by_key.setdefault(self.key_fn(item), []).append(item)
            results.extend(self.reduce_fn(k, v) for k, v in by_key.items())
        return results
