"""D — sleep/idle validation and timeout watch."""
import asyncio
from unittest.mock import MagicMock, patch

import pytest

from ray_hive.errors import ConfigError
from ray_hive.hive import _split_vllm_kwargs


def test_split_vllm_kwargs_lifts_planner_keys():
    overrides, rest = _split_vllm_kwargs({"max_num_seqs": 8, "temperature": 0.5})
    assert overrides == {"max_num_seqs": 8}
    assert rest == {"temperature": 0.5}


def test_idle_sleep_validation():
    def check(idle, sleep):
        if idle != -1 and idle <= 0:
            raise ConfigError("idle")
        if sleep != -1 and sleep <= 0:
            raise ConfigError("sleep")
        if idle > 0 and sleep > 0 and idle <= sleep:
            raise ConfigError("order")

    check(-1, -1)
    check(30, 10)
    with pytest.raises(ConfigError):
        check(0, -1)
    with pytest.raises(ConfigError):
        check(10, 10)


@pytest.mark.asyncio
async def test_timeout_watch_idle_destroys():
    from ray_hive.core.model_router import ModelRouter

    router = object.__new__(ModelRouter.func_or_class)
    router.model_id = "m"
    router.idle_timeout = 10
    router.sleep_timeout = -1
    router._shutting_down = False
    router._sleeping = False
    router._sleep_lock = asyncio.Lock()
    router.gpu_deployment_names = []
    router._last_activity = 0.0
    router._get_handles = lambda: {}

    shutdown = MagicMock()

    async def instant_sleep(_interval):
        return None

    async def fake_to_thread(fn, *a, **k):
        return fn(*a, **k)

    with patch("time.time", return_value=100.0), \
         patch("asyncio.sleep", side_effect=instant_sleep), \
         patch("asyncio.to_thread", side_effect=fake_to_thread), \
         patch("ray_hive.core.ray_utils.lifecycle.shutdown_model", shutdown):
        await router._timeout_watch()

    assert router._shutting_down
    shutdown.assert_called_once_with("m")


def test_touch_updates_activity():
    from ray_hive.core.model_router import ModelRouter

    router = object.__new__(ModelRouter.func_or_class)
    router._last_activity = 0.0
    router._touch()
    assert router._last_activity > 0


@pytest.mark.asyncio
async def test_timeout_watch_holds_vram_before_sleep():
    from ray_hive.core.model_router import ModelRouter

    order = []
    sleep_handle = MagicMock()

    router = object.__new__(ModelRouter.func_or_class)
    router.model_id = "m"
    router.idle_timeout = -1
    router.sleep_timeout = 10
    router._shutting_down = False
    router._sleeping = False
    router._sleep_lock = asyncio.Lock()
    router.gpu_deployment_names = ["r1"]
    router._last_activity = 0.0
    router._get_handles = lambda: {"r1": sleep_handle}

    async def hold():
        order.append("mark_sleeping")

    async def fake_gather(*_a, **_k):
        order.append("engine_sleep")
        router._shutting_down = True

    async def instant_sleep(_interval):
        return None

    with patch.object(router, "_hold_vram_for_sleep", side_effect=hold), \
         patch("time.time", return_value=100.0), \
         patch("asyncio.sleep", side_effect=instant_sleep), \
         patch("asyncio.gather", side_effect=fake_gather):
        await router._timeout_watch()

    assert router._sleeping
    assert order == ["mark_sleeping", "engine_sleep"]


@pytest.mark.asyncio
async def test_ensure_awake_releases_vram_after_wake():
    from ray_hive.core.model_router import ModelRouter

    order = []
    wake_handle = MagicMock()

    router = object.__new__(ModelRouter.func_or_class)
    router.sleep_timeout = 10
    router._sleeping = True
    router._sleep_lock = asyncio.Lock()
    router.gpu_deployment_names = ["r1"]
    router._get_handles = lambda: {"r1": wake_handle}

    async def release():
        order.append("mark_awake")

    async def fake_gather(*_a, **_k):
        order.append("engine_wake")

    with patch.object(router, "_release_vram_after_wake", side_effect=release), \
         patch("asyncio.gather", side_effect=fake_gather):
        await router._ensure_awake()

    assert not router._sleeping
    assert order == ["engine_wake", "mark_awake"]
