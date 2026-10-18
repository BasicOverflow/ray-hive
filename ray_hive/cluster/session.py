"""Project-scoped Ray connect with banner."""
from __future__ import annotations

import os
from typing import Any

from ray_hive.cluster.banner import print_hive_banner
from ray_hive.cluster.project import ProjectContext
from ray_hive.cluster.registry import ClaimKind, get_cluster_registry, register_project
from ray_hive.core.ray_utils.naming import NAMESPACE_ENV
from ray_hive.core.ray_utils.session import init_ray


def init_ray_for_project(
    address: str,
    ctx: ProjectContext,
    *,
    suppress_logging: bool = True,
    register: bool = True,
    claim_route: bool = True,
    **kwargs: Any,
):
    """``init_ray`` into ``ctx.namespace``, print GitHub banner, optionally register claims."""
    print_hive_banner()
    kwargs.setdefault("namespace", ctx.namespace)
    os.environ[NAMESPACE_ENV] = ctx.namespace
    os.environ.setdefault("RAY_HIVE_PROJECT_ID", ctx.project_id)
    init_ray(address, suppress_logging=suppress_logging, **kwargs)
    if register:
        register_project(ctx)
        if claim_route and ctx.serve_route_prefix:
            import ray

            reg = get_cluster_registry()
            ray.get(
                reg.claim.remote(
                    ClaimKind.SERVE_ROUTE.value,
                    ctx.serve_route_prefix,
                    ctx.project_id,
                )
            )
    return ctx
