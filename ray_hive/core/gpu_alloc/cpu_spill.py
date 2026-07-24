"""TP=1 host RAM extension — weight spill and CPU KV (allocator layer)."""
_GPU_BUDGET_FRAC = 0.97


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
    """-1 = 50% of host memory, split across replicas on that host."""
    if cfg < -1:
        raise ValueError(f"cpu_ram_per_instance must be >= -1, got {cfg}")
    if cfg == -1:
        if host_memory_gb <= 0:
            raise ValueError(
                f"cpu_ram_per_instance=-1: host memory budget is {host_memory_gb:.2f}GB"
            )
        n = max(1, int(replicas_on_host))
        return float(0.50 * host_memory_gb / n)
    if cfg < 0:
        raise ValueError(f"cpu_ram_per_instance must be 0, -1, or >0, got {cfg}")
    return float(cfg)


def resolve_cpu_spill(
    cpu_ram_gb: float,
    weight_need_gb: float,
    gpu_budget_gb: float,
) -> tuple[float, float]:
    """Split host budget: weights on GPU if they fit; else spill; remainder → KV."""
    if cpu_ram_gb <= 0:
        return 0.0, 0.0
    if weight_need_gb <= gpu_budget_gb:
        return 0.0, float(cpu_ram_gb)
    cpu_offload = weight_need_gb - gpu_budget_gb
    if cpu_offload > cpu_ram_gb:
        raise ValueError(
            f"Weight spill needs {cpu_offload:.2f}GB host RAM, "
            f"but cpu_ram_per_instance budget is {cpu_ram_gb:.2f}GB"
        )
    return float(cpu_offload), float(cpu_ram_gb - cpu_offload)


def on_gpu_weight_need_gb(
    weight_need: float,
    available_gb: float,
    cpu_ram_cfg: float,
    host_memory_gb: float,
    replicas_on_host: int = 1,
) -> float:
    """On-GPU GiB required after optional host weight spill (TP=1 only)."""
    gpu_budget = available_gb * _GPU_BUDGET_FRAC
    cpu_ram = resolve_cpu_ram_budget(cpu_ram_cfg, host_memory_gb, replicas_on_host)
    cpu_offload, _ = resolve_cpu_spill(cpu_ram, weight_need, gpu_budget)
    return weight_need - cpu_offload


def resolve_group_cpu_spill(
    weight_need_gb: float,
    available_gb: float,
    cpu_ram_cfg: float,
    host_memory_gb: float,
    replicas_on_host: int,
) -> tuple[float, float, float, float, int]:
    """TP=1 per-group GPU budget + host spill/KV split + Ray memory claim."""
    per_gpu_budget = available_gb * _GPU_BUDGET_FRAC
    cpu_ram = resolve_cpu_ram_budget(cpu_ram_cfg, host_memory_gb, replicas_on_host)
    cpu_offload, kv_offload = resolve_cpu_spill(cpu_ram, weight_need_gb, per_gpu_budget)
    memory_bytes = int((cpu_offload + kv_offload) * 1024 ** 3)
    return per_gpu_budget, cpu_ram, cpu_offload, kv_offload, memory_bytes
