"""Meta-supervisor heal: system actors, GC, sticky models, collision refuse."""
from __future__ import annotations

import logging
import time

import ray
from ray import serve

from ray_hive.cluster.registry import get_cluster_registry
from ray_hive.core.deployment import get_deploy_service
from ray_hive.core.gpu_registry import get_gpu_registry
from ray_hive.core.openai_gateway import ensure_openai_gateway
from ray_hive.core.ray_utils.lifecycle import reconcile_with_serve
from ray_hive.jobs.admission import get_admission_broker
from ray_hive.jobs.cron import get_cron_board
from ray_hive.supervisor.desired_state import DesiredState, StickyCron, StickyProject

logger = logging.getLogger("ray_hive.supervisor.heal")


def ensure_system_actors() -> None:
    get_cluster_registry()
    get_admission_broker()
    get_cron_board()


def ensure_actors() -> None:
    ensure_system_actors()
    try:
        get_gpu_registry()
        get_deploy_service()
    except Exception:
        logger.warning("vLLM actors unavailable (install ray-hive[vllm]?)", exc_info=True)


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
    try:
        registry = get_gpu_registry()
        apps = set((serve.status().applications or {}).keys())
        n = reconcile_with_serve(registry, apps)
        if n:
            logger.info("reconcile_with_serve cleared %s ghost entries", n)
        return n
    except Exception:
        logger.warning("reconcile skipped", exc_info=True)
        return 0


def live_model_ids() -> set[str]:
    try:
        registry = get_gpu_registry()
        return set(ray.get(registry.list_model_ids.remote()) or [])
    except Exception:
        return set()


def sync_meta_from_cluster(state: DesiredState) -> None:
    """Pull live registry/cron board into SQLite; GC stale projects."""
    try:
        reg = get_cluster_registry()
        for p in ray.get(reg.list_projects.remote()):
            state.upsert_project(
                StickyProject(
                    project_id=p["project_id"],
                    namespace=p["namespace"],
                    serve_route_prefix=p.get("serve_route_prefix"),
                    model_id_prefix=p.get("model_id_prefix"),
                    scratch_prefix=p.get("scratch_prefix"),
                    last_heartbeat_at=float(p.get("last_heartbeat_at") or time.time()),
                    state=p.get("state") or "alive",
                )
            )
        gc = ray.get(reg.gc.remote())
        if gc.get("purged") or gc.get("dead"):
            logger.info("cluster registry gc %s", gc)
    except Exception:
        logger.warning("registry sync failed", exc_info=True)
    try:
        board = get_cron_board()
        for e in ray.get(board.list_entries.remote()):
            state.upsert_cron(
                StickyCron(
                    cron_id=e["cron_id"],
                    project_id=e["project_id"],
                    schedule=e["schedule"],
                    tz=e.get("tz") or "America/New_York",
                    load_class=e.get("load_class") or "cpu",
                    enabled=bool(e.get("enabled", True)),
                    next_fire=e.get("next_fire"),
                    last_fire=e.get("last_fire"),
                    last_status=e.get("last_status"),
                )
            )
    except Exception:
        logger.warning("cron board sync failed", exc_info=True)
    dead = state.gc_stale_projects()
    if dead:
        logger.info("purged stale projects %s", dead)
        try:
            board = get_cron_board()
            for pid in dead:
                ray.get(board.remove_project.remote(pid))
        except Exception:
            pass


def _route_claimed_by_other(model_id: str, project_id: str | None) -> bool:
    """Refuse sticky redeploy if model_id claim belongs to another project."""
    try:
        reg = get_cluster_registry()
        for c in ray.get(reg.list_claims.remote()):
            if c.get("kind") == "model_id" and c.get("name") == model_id:
                holder = c.get("project_id")
                if project_id and holder and holder != project_id:
                    return True
                if not project_id and holder and holder not in ("ray_hive", None):
                    return True
    except Exception:
        return False
    return False


def redeploy_sticky(state: DesiredState, *, cool_down_ok: bool) -> list[str]:
    if not cool_down_ok:
        return []
    live = live_model_ids()
    try:
        apps = set((serve.status().applications or {}).keys())
    except Exception:
        apps = set()
    attempted = []
    for model in state.list():
        if model.model_id in live and model.model_id in apps:
            continue
        if _route_claimed_by_other(model.model_id, model.project_id):
            logger.warning(
                "skip sticky redeploy collision model_id=%s project=%s",
                model.model_id,
                model.project_id,
            )
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
    ensure_system_actors()
    sync_meta_from_cluster(state)
    try:
        ensure_actors()
        wait_vram_settled()
        reconcile_registry()
    except Exception:
        logger.warning("vLLM heal path skipped", exc_info=True)
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
        try:
            redeploy_sticky(state, cool_down_ok=True)
        except Exception:
            logger.warning("sticky redeploy skipped", exc_info=True)
