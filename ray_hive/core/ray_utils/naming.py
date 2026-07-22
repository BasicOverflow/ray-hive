"""Serve / Ray resource naming for model replicas and GPU pins."""


def gpu_resource_name(gpu_key: str) -> str:
    """Return Ray custom resource name for a GPU key (node_gpuN)."""
    node_name, gpu_id_str = gpu_key.split(":")
    gpu_id = gpu_id_str.replace("gpu", "")
    return f"{node_name}_gpu{gpu_id}"


def deployment_name(model_id: str, gpu_keys: list[str]) -> str:
    """Return Serve app/deployment name for a model replica on one or more GPUs."""
    if len(gpu_keys) == 1:
        return f"{model_id}-{gpu_keys[0].replace(':', '-').replace('_', '-')}"
    host = gpu_keys[0].split(":")[0].replace("_", "-")
    ids = "-".join(k.split(":")[1] for k in gpu_keys)
    return f"{model_id}-{host}-{ids}"


def gpu_info_entry(gpu_key: str, gpu_info: dict) -> dict:
    """Build scheduling info dict from a registry GPU view."""
    return {
        "gpu_key": gpu_key,
        "resource_name": gpu_resource_name(gpu_key),
        "gpu_id": gpu_key.split(":")[1].replace("gpu", ""),
        "total_gb": gpu_info["total"],
        "free_gb": gpu_info["free"],
        "available_gb": gpu_info["available"],
    }
