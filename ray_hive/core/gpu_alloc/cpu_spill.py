"""TP=1 host RAM extension — weight spill only (allocator layer)."""

# Packing fraction for TP=1 (must match gpu_budget_frac(1) in placement).
# Leave ~10% outside the util pool for CUDA-graph capture + sampler scratch
# (vLLM's own default gpu_memory_utilization is 0.9 for the same reason).
TP1_BUDGET_FRAC = 0.90
# Auto cpu_ram_per_instance=-1 ceiling: share of Ray free host memory / replicas.
HOST_RAM_AUTO_FRAC = 0.70

_KV_OFFLOAD_KWARGS = (
    "kv_offloading_size",
    "kv_offloading_backend",
    "kv_transfer_config",
)


def reject_unsupported_host_ram_kwargs(vllm_kwargs: dict) -> None:
    """Hive owns host RAM; reject user KV / weight-offload engine kwargs."""
    found = [k for k in _KV_OFFLOAD_KWARGS if k in vllm_kwargs]
    if found:
        raise ValueError(
            f"KV offloading is not supported yet (got {', '.join(found)}). "
            "Use cpu_ram_per_instance for host weight spill only."
        )
    if "cpu_offload_gb" in vllm_kwargs:
        raise ValueError(
            "Do not pass cpu_offload_gb as a vLLM kwarg; "
            "use cpu_ram_per_instance for host weight spill."
        )


def assert_cpu_ram_tp_allowed(tp_size: int, cpu_ram_cfg: float) -> None:
    if tp_size > 1 and cpu_ram_cfg != 0:
        raise ValueError(
            "cpu_ram_per_instance is not supported with tensor parallelism (TP>1); "
            "use 0 and fit on GPU VRAM only"
        )


def resolve_cpu_ram_budget(
    cfg: float,
    host_memory_gb: float,
    replicas_on_host: int = 1,
) -> float:
    """Max host GiB this replica may use for weight spill.

    -1 → HOST_RAM_AUTO_FRAC of Ray free host memory / replicas_on_host (ceiling;
    actual use is only the weight spill needed — see resolve_group_cpu_spill).
    >0 → hard GiB, capped by that host share when known.
    """
    if cfg < -1:
        raise ValueError(f"cpu_ram_per_instance must be >= -1, got {cfg}")
    per_replica_host = float(host_memory_gb) / int(replicas_on_host) if host_memory_gb > 0 else 0.0
    if cfg == -1:
        if per_replica_host <= 0:
            raise ValueError(
                f"cpu_ram_per_instance=-1: host memory budget is {host_memory_gb:.2f}GB"
            )
        return float(HOST_RAM_AUTO_FRAC * per_replica_host)
    if cfg < 0:
        raise ValueError(f"cpu_ram_per_instance must be 0, -1, or >0, got {cfg}")
    if cfg > 0 and per_replica_host > 0:
        return float(min(cfg, per_replica_host))
    return float(cfg)


def resolve_cpu_spill(
    cpu_ram_gb: float,
    weight_need_gb: float,
    gpu_budget_gb: float,
) -> tuple[float, float]:
    """Host budget → weight spill only. Leftover is unused (no CPU KV connector)."""
    if cpu_ram_gb <= 0:
        return 0.0, 0.0
    if weight_need_gb <= gpu_budget_gb:
        return 0.0, 0.0
    cpu_offload = weight_need_gb - gpu_budget_gb
    if cpu_offload > cpu_ram_gb:
        raise ValueError(
            f"Weight spill needs {cpu_offload:.2f}GB host RAM, "
            f"but cpu_ram_per_instance budget is {cpu_ram_gb:.2f}GB"
        )
    return float(cpu_offload), 0.0


def on_gpu_weight_need_gb(
    weight_need: float,
    available_gb: float,
    cpu_ram_cfg: float,
    host_memory_gb: float,
    replicas_on_host: int = 1,
) -> float:
    """On-GPU GiB required after optional host weight spill (TP=1 only)."""
    gpu_budget = available_gb * TP1_BUDGET_FRAC
    ceiling = resolve_cpu_ram_budget(cpu_ram_cfg, host_memory_gb, replicas_on_host)
    cpu_offload, _ = resolve_cpu_spill(ceiling, weight_need, gpu_budget)
    return weight_need - cpu_offload


def resolve_group_cpu_spill(
    weight_need_gb: float,
    available_gb: float,
    cpu_ram_cfg: float,
    host_memory_gb: float,
    replicas_on_host: int,
) -> tuple[float, float, float, float, int]:
    """TP=1 per-group GPU budget + host weight spill.

    Returns (per_gpu_budget, cpu_ram_budget, cpu_offload, kv_offload, memory_bytes).
    For cfg=-1, cpu_ram_budget equals the weight spill needed (0 if weights fit).
    For cfg>0, cpu_ram_budget is the hard ceiling; cpu_offload is what is used.
    """
    per_gpu_budget = available_gb * TP1_BUDGET_FRAC
    ceiling = resolve_cpu_ram_budget(cpu_ram_cfg, host_memory_gb, replicas_on_host)
    cpu_offload, kv_offload = resolve_cpu_spill(ceiling, weight_need_gb, per_gpu_budget)
    cpu_ram = float(cpu_offload) if cpu_ram_cfg == -1 else ceiling
    memory_bytes = int((cpu_offload + kv_offload) * 1024 ** 3)
    return per_gpu_budget, cpu_ram, cpu_offload, kv_offload, memory_bytes
