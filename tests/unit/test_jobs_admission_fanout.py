"""AdmissionBrokerLogic window + BatchFanout with mocked ray."""
from unittest.mock import MagicMock, patch

from ray_hive.jobs.admission import AdmissionBrokerLogic, TaskShape
from ray_hive.jobs.fanout import chunks, run_batches


def test_chunks():
    assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_admit_respects_ceiling():
    broker = AdmissionBrokerLogic()
    with patch.object(broker, "_cluster_cpus", return_value=100.0):
        broker._window = 50
        n = broker.admit(TaskShape(max_ceiling=4, cpus_per_task=1.0))
        assert n == 4


def test_note_failure_shrinks():
    broker = AdmissionBrokerLogic()
    broker._window = 16
    assert broker.note_failure("oom") == 8


def test_run_batches_drains():
    shape = TaskShape(batch_size=2, max_in_flight=2, max_ceiling=8)

    def submit(batch):
        ref = MagicMock()
        ref._batch = batch
        return ref

    def fake_wait(pending, num_returns=1):
        r = pending[0]
        return [r], pending[1:]

    def fake_get(ref):
        return sum(ref._batch)

    with patch("ray_hive.jobs.fanout.ray") as fake_ray:
        fake_ray.wait.side_effect = fake_wait
        fake_ray.get.side_effect = fake_get
        out = run_batches("t", [1, 2, 3, 4], submit, shape, use_broker=False)
    assert out == [3, 7]
