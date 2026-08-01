"""Live Deploy cycle fixtures — session RayHive + small-model constants."""
import os
import uuid

import pytest

TEXT_MODEL = "Qwen/Qwen3-0.6B-FP8"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# Smallest MM commonly used in examples; skip cycle if VRAM claim fails.
MM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

VLLM_TEXT = dict(
    max_num_seqs=4,
    trust_remote_code=True,
    reasoning_parser="qwen3",
    default_chat_template_kwargs={"enable_thinking": False},
)


@pytest.fixture(scope="session")
def hive():
    addr = os.environ.get("RAY_ADDRESS")
    if not addr:
        pytest.skip("RAY_ADDRESS not set")
    from ray_hive import RayHive

    h = RayHive(address=addr, suppress_logging=True)
    yield h


@pytest.fixture(scope="session")
def scheduler(hive):
    from tests.live.cluster_sched import ClusterScheduler

    return ClusterScheduler(hive.get_vram_state, poll_s=3.0, max_wait_s=240.0)


def uniq(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"
