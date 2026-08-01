"""
Ray utility helpers — session, hardware, naming, placement, GPU selection, TP,
lifecycle, media I/O, multimodal token sizing.

Import from this package (``from ray_hive.core.ray_utils import sm_count``) or
from the typed submodules (``from ray_hive.core.ray_utils.hardware import sm_count``).
"""
from .hardware import (
    approx_tdp,
    compute_cap,
    count_by_host,
    filter_alive,
    gpu_inventory_lines,
    host_memory_available_gb,
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
from .media import (
    audio_array_from_bytes,
    audio_from_b64,
    audio_from_url,
    file_to_data_url,
    load_bytes_from_url,
    pil_from_url,
    video_frames_from_url,
)
from .mm_helpers import (
    audio_tokens_per_item,
    estimate_encoder_params,
    mm_count,
    mm_size_hint,
    vision_tokens_per_image,
    vision_tokens_per_video,
)
from .naming import deployment_name, gpu_info_entry, gpu_resource_name
from .placement import (
    build_vram_reqs_for_tp,
    chunk_gpu_groups,
    fixed_non_kv_gb,
    plan_replica_groups,
)
from .select_gpus import resolve_target_gpus
from .session import StderrFilter, init_ray, serve_base_url, suppress_ray_warnings
from .tensor_parallel import assert_tp_shardable, tp_shardable
from .display import (
    error,
    info,
    print_banner,
    print_deployment_plan,
    print_panel,
    success,
    warn,
)

__all__ = [
    "StderrFilter",
    "approx_tdp",
    "assert_model_id_free",
    "assert_tp_shardable",
    "audio_array_from_bytes",
    "audio_from_b64",
    "audio_from_url",
    "audio_tokens_per_item",
    "build_vram_reqs_for_tp",
    "chunk_gpu_groups",
    "compute_cap",
    "count_by_host",
    "deployment_name",
    "error",
    "estimate_encoder_params",
    "file_to_data_url",
    "filter_alive",
    "fixed_non_kv_gb",
    "gpu_info_entry",
    "gpu_inventory_lines",
    "gpu_resource_name",
    "host_memory_available_gb",
    "info",
    "init_ray",
    "is_node_alive",
    "kill_gpu_registry",
    "load_bytes_from_url",
    "max_gpus_on_any_host",
    "mem_bandwidth",
    "mm_count",
    "mm_size_hint",
    "pil_from_url",
    "plan_replica_groups",
    "print_banner",
    "print_deployment_plan",
    "print_panel",
    "resolve_target_gpus",
    "serve_base_url",
    "shutdown_all",
    "shutdown_model",
    "sm_count",
    "success",
    "suppress_ray_warnings",
    "tp_shardable",
    "video_frames_from_url",
    "vision_tokens_per_image",
    "vision_tokens_per_video",
    "warn",
]
