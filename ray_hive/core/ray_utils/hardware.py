"""GPU hardware readers from registry PyCUDA specs + Alive-node helpers."""
import ray


# Approximate board TGP/TDP (W) by substring match on nvidia-smi / PyCUDA name.
# Values from NVIDIA product pages / datasheets (Founders / reference TGP).
# approx_tdp() matches longest key first so "4070 ti super" wins over "4070".
_TDP_BY_NAME = {
    # GeForce RTX 50 (Blackwell)
    "5090": 575,
    "5080": 360,
    "5070 ti": 300,
    "5070": 250,
    "5060 ti": 180,
    "5060": 145,
    "5050": 130,
    # GeForce RTX 40 (Ada)
    "4090": 450,
    "4080 super": 320,
    "4080": 320,
    "4070 ti super": 285,
    "4070 ti": 285,
    "4070 super": 220,
    "4070": 200,
    "4060 ti": 160,
    "4060": 115,
    "4050": 115,
    # GeForce RTX 30 (Ampere)
    "3090 ti": 450,
    "3090": 350,
    "3080 ti": 350,
    "3080": 320,
    "3070 ti": 290,
    "3070": 220,
    "3060 ti": 200,
    "3060": 170,
    "3050": 130,
    # Workstation / datacenter
    "rtx 6000 ada": 300,
    "rtx 5000 ada": 250,
    "rtx 4000 ada": 130,
    "a6000": 300,
    "a5500": 230,
    "a5000": 230,
    "a4500": 200,
    "a4000": 140,
    "a2000": 70,
    "l40s": 350,
    "l40": 300,
    "l4": 72,
    "a100": 400,
    "h100": 700,
    "t4": 70,
}


def sm_count(gpu: dict) -> int:
    """Return multiprocessor_count from registry PyCUDA specs."""
    return int(gpu["specs"]["multiprocessor_count"])


def compute_cap(gpu: dict) -> tuple[int, int]:
    """Return (major, minor) compute capability from registry PyCUDA specs."""
    specs = gpu["specs"]
    return (int(specs["compute_capability_major"]), int(specs["compute_capability_minor"]))


def approx_tdp(gpu: dict) -> float:
    """Return approximate TDP watts from GPU name (static map, longest key first)."""
    name = str(gpu.get("specs", {}).get("name", "")).lower()
    for key, watts in sorted(_TDP_BY_NAME.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key in name:
            return float(watts)
    return 250.0


def mem_bandwidth(gpu: dict) -> float:
    """Return memory bandwidth proxy: bus_width * memory_clock_rate."""
    specs = gpu["specs"]
    return float(specs["global_memory_bus_width"]) * float(specs["memory_clock_rate"])


def _node_matches_hostname(node: dict, hostname: str) -> bool:
    """True when a Ray node dict corresponds to a registry hostname."""
    if node.get("NodeManagerHostname") == hostname or node.get("NodeName") == hostname:
        return True
    resources = node.get("Resources") or {}
    return any(key.startswith(f"{hostname}_gpu") for key in resources)


def is_node_alive(hostname: str) -> bool:
    """True when an Alive Ray node matches the registry hostname."""
    for node in ray.nodes():
        if node.get("Alive") and _node_matches_hostname(node, hostname):
            return True
    return False


def host_memory_available_gb(hostname: str) -> float:
    """Ray logical memory still free (GiB) for a registry hostname."""
    for node in ray.nodes():
        if not node.get("Alive") or not _node_matches_hostname(node, hostname):
            continue
        # Prefer currently free memory; fall back to capacity.
        resources = node.get("AvailableResources") or node.get("Resources") or {}
        return float(resources.get("memory", 0)) / (1024 ** 3)
    return 0.0


def filter_alive(eligible: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Drop GPUs whose registry host is not an Alive Ray node."""
    return [(k, g) for k, g in eligible if is_node_alive(k.split(":")[0])]


def gpu_inventory_lines(gpu_map: dict) -> str:
    """One-line-per-GPU inventory string for placement errors."""
    lines = []
    for k, g in sorted(gpu_map.items()):
        try:
            cap = compute_cap(g)
            cap_s = f"sm{cap[0]}{cap[1]}"
        except (KeyError, TypeError, ValueError):
            cap_s = "sm?"
        lines.append(
            f"  {k}: avail={g.get('available', 0):.2f}GB / total={g.get('total', 0):.2f}GB "
            f"({cap_s})"
        )
    return "\n".join(lines)


def count_by_host(gpu_keys) -> dict[str, int]:
    """Count gpu keys (or any 'host:...' strings) per hostname."""
    counts: dict[str, int] = {}
    for key in gpu_keys:
        h = key.split(":")[0]
        counts[h] = counts.get(h, 0) + 1
    return counts


def max_gpus_on_any_host(gpu_map: dict) -> int:
    """Largest same-host GPU count in the registry."""
    counts = count_by_host(gpu_map)
    return max(counts.values()) if counts else 0
