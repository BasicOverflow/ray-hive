"""Cluster identity, collision claims, and project-scoped Ray connect."""
from .banner import print_hive_banner
from .project import ProjectContext
from .registry import ClaimKind, ClusterRegistry, ClusterRegistryLogic, get_cluster_registry

__all__ = [
    "ProjectContext",
    "ClusterRegistry",
    "ClusterRegistryLogic",
    "ClaimKind",
    "get_cluster_registry",
    "print_hive_banner",
    "init_ray_for_project",
]


def __getattr__(name):
    if name == "init_ray_for_project":
        from .session import init_ray_for_project

        return init_ray_for_project
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
