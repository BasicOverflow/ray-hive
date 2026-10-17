"""Heal hive actors, VRAM settle, Serve reconcile, sticky redeploy."""
from __future__ import annotations

import logging
import time

import ray
from ray import serve

from ray_hive.core.deployment import get_deploy_service
from ray_hive.core.gpu_registry import get_gpu_registry
from ray_hive.core.openai_gateway import ensure_openai_gateway
from ray_hive.core.ray_utils.lifecycle import reconcile_with_serve
from ray_hive.supervisor.desired_state import DesiredState

logger = logging.getLogger("ray_hive.supervisor.heal")


def ensure_actors() -> None:
    get_gpu_registry()
    get_deploy_service()


def wait_vram_settled(timeout_s: float = 90.0) -> dict:
    registry = get_gpu_registry()
    deadline = time.time() + timeout_s
    last: dict = {}
    while True:
        last = ray.get(registry.get_all_gpus.remote())
        if last and any(float(g.get("free") or 0) > 0 for g in last.values()):
            return last
        if time.time() >= deadline:
            raise TimeoutError(
                f"VRAM registry empty/zero after {timeout_s:.0f}s keys={list(last)}"
            )
        time.sleep(1.0)


def reconcile_registry() -> int:
    registry = get_gpu_registry()
    apps = set((serve.status().applications or {}).keys())
    n = reconcile_with_serve(registry, apps)
    if n:
        logger.info("reconcile_with_serve cleared %s ghost entries", n)
    return n


def live_model_ids() -> set[str]:
    registry = get_gpu_registry()
    return set(ray.get(registry.list_model_ids.remote()) or [])


def redeploy_sticky(state: DesiredState, *, cool_down_ok: bool) -> list[str]:
    """Redeploy missing sticky models. Returns model_ids attempted."""
    if not cool_down_ok:
        return []
    live = live_model_ids()
    apps = set((serve.status().applications or {}).keys())
    attempted = []
    for model in state.list():
        if model.model_id in live and model.model_id in apps:
            continue
        attempted.append(model.model_id)
        logger.info("redeploy sticky model_id=%s name=%s", model.model_id, model.model_name)
        try:
            svc = get_deploy_service()
            config = {
                "name": model.model_name,
                "replicas": model.replicas,
                "gpu": model.gpu,
                "max_input_prompt_length": model.max_input_prompt_length,
                "max_output_prompt_length": model.max_output_prompt_length,
                "idle_timeout": model.idle_timeout,
                "sleep_timeout": model.sleep_timeout,
            }
            ray.get(
                svc.deploy_models.remote(
                    {model.model_id: config},
                    {model.model_id: dict(model.vllm_kwargs or {})},
                )
            )
            ensure_openai_gateway()
        except Exception:
            logger.exception("sticky redeploy failed model_id=%s", model.model_id)
    return attempted


def heal_once(state: DesiredState, *, attempt_serve: bool) -> None:
    ensure_actors()
    wait_vram_settled()
    reconcile_registry()
    if attempt_serve:
        try:
            serve.start(
                proxy_location="HeadOnly",
                http_options={"host": "0.0.0.0", "port": 8000},
            )
        except Exception:
            logger.warning("serve.start skipped / already up", exc_info=True)
        try:
            ensure_openai_gateway()
        except Exception:
            logger.warning("gateway ensure failed", exc_info=True)
        redeploy_sticky(state, cool_down_ok=True)
