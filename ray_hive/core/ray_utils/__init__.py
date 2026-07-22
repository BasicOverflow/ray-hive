"""
Ray utility helpers — session, hardware, naming, placement, TP, lifecycle.

Import from this package (``from ray_hive.core.ray_utils import sm_count``) or
from the typed submodules (``from ray_hive.core.ray_utils.hardware import sm_count``).
"""
from .hardware import (
    approx_tdp,
    compute_cap,
    filter_alive,
    gpu_inventory_lines,
    is_node_alive,
    max_gpus_on_any_host,
    mem_bandwidth,
    sm_count,
)
from .lifecycle import (
    assert_model_id_free,
    kill_gpu_registry,
    shutdown_all,
    shutdown_model,
)
from .naming import deployment_name, gpu_info_entry, gpu_resource_name
from .placement import (
    build_vram_reqs_for_tp,
    chunk_gpu_groups,
    fixed_non_kv_gb,
    resolve_cpu_ram_budget,
    resolve_cpu_spill,
    resolve_pinned_gpu,
)
from .session import StderrFilter, init_ray, suppress_ray_warnings
from .tensor_parallel import assert_tp_shardable, tp_shardable

__all__ = [
    "StderrFilter",
    "approx_tdp",
    "assert_model_id_free",
    "assert_tp_shardable",
    "build_vram_reqs_for_tp",
    "chunk_gpu_groups",
    "compute_cap",
    "deployment_name",
    "filter_alive",
    "fixed_non_kv_gb",
    "gpu_info_entry",
    "gpu_inventory_lines",
    "gpu_resource_name",
    "init_ray",
    "is_node_alive",
    "kill_gpu_registry",
    "max_gpus_on_any_host",
    "mem_bandwidth",
    "resolve_cpu_ram_budget",
    "resolve_cpu_spill",
    "resolve_pinned_gpu",
    "shutdown_all",
    "shutdown_model",
    "sm_count",
    "suppress_ray_warnings",
    "tp_shardable",
]
