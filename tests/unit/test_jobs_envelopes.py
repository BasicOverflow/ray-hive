"""Envelope options always force worker label."""
from ray_hive.jobs.envelopes import (
    WORKER_LABEL_SELECTOR,
    cpu_options,
    io_options,
    ops_options,
)


def test_io_forces_label():
    opts = io_options(label_selector={"evil": "yes"}, num_cpus=0.5)
    assert opts["label_selector"] == WORKER_LABEL_SELECTOR
    assert opts["num_cpus"] == 0.5


def test_cpu_broker_sentinel():
    opts = cpu_options(num_cpus=-1)
    assert opts.get("_broker_cpus") is True
    assert opts["num_cpus"] == 1
    assert opts["label_selector"] == WORKER_LABEL_SELECTOR


def test_ops_no_worker_label():
    opts = ops_options()
    assert "label_selector" not in opts
