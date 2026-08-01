"""M — estimate helpers + async API guard."""
import asyncio

import pytest

from ray_hive.hive import _split_vllm_kwargs
from ray_hive.inference import _assert_not_in_asyncio_loop


def test_split_preserves_engine_kwargs():
    o, r = _split_vllm_kwargs({"max_num_batched_tokens": 512, "trust_remote_code": True})
    assert o["max_num_batched_tokens"] == 512
    assert r["trust_remote_code"] is True


@pytest.mark.asyncio
async def test_sync_api_blocked_in_loop():
    with pytest.raises(AssertionError):
        _assert_not_in_asyncio_loop()


def test_sync_api_ok_outside_loop():
    _assert_not_in_asyncio_loop()
