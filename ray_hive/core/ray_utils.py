"""Ray utility functions with warning suppression."""
import logging
import os
import re
import sys
import warnings

import ray
from dotenv import load_dotenv


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_env():
    """Load .env from project root if present."""
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))


load_env()


class StderrFilter:
    """Filters stderr to suppress Ray C++ warnings."""

    def __init__(self, original_stderr):
        """Store original stderr and suppression pattern list."""
        self.original_stderr = original_stderr
        self.suppress_patterns = [
            "Python patch version mismatch",
            "Failed to connect to GCS",
            "Timed out while waiting for GCS",
            "Failed to get queue length",
            "LongPollClient connection failed",
            "SIGTERM handler is not set",
            "rpc_client.h",
            "gcs_client.cc",
            "InvalidStateError: CANCELLED",
            "InvalidStateError",
            "concurrent.futures._base.InvalidStateError",
            "state=cancelled",
            "Callback error",
            "dataclient.py",
            "ray_client_streaming_rpc",
        ]


    def write(self, text):
        """Write to stderr unless text matches a suppressed Ray noise pattern."""
        if not text.strip():
            self.original_stderr.write(text)
            return
        if re.match(r'\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', text.strip()):
            if " W " in text or "INFO" in text or "WARNING" in text:
                return
        if any(pattern in text for pattern in self.suppress_patterns):
            return
        self.original_stderr.write(text)


    def flush(self):
        """Flush the underlying stderr stream."""
        self.original_stderr.flush()


def suppress_ray_warnings(suppress: bool = True):
    """Suppress Ray warnings and logs only, preserving user print statements."""
    if suppress:
        warnings.filterwarnings("ignore", module="ray.*")
        warnings.filterwarnings("ignore", message=".*ray.*", category=Warning)
        logging.getLogger("ray").setLevel(logging.CRITICAL)
        logging.getLogger("ray.serve").setLevel(logging.CRITICAL)
        logging.getLogger("ray.util").setLevel(logging.CRITICAL)
        logging.getLogger("ray.data").setLevel(logging.CRITICAL)
        logging.getLogger("ray.train").setLevel(logging.CRITICAL)
        os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
        os.environ["RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S"] = "10"
        os.environ["RAY_SCHEDULER_EVENTS"] = "0"
        if not isinstance(sys.stderr, StderrFilter):
            sys.stderr = StderrFilter(sys.stderr)
    else:
        warnings.filterwarnings("default", module="ray.*")
        warnings.filterwarnings("default", message=".*ray.*", category=Warning)
        logging.getLogger("ray").setLevel(logging.INFO)
        logging.getLogger("ray.serve").setLevel(logging.INFO)
        logging.getLogger("ray.util").setLevel(logging.INFO)
        logging.getLogger("ray.data").setLevel(logging.INFO)
        logging.getLogger("ray.train").setLevel(logging.INFO)
        os.environ.pop("RAY_DISABLE_IMPORT_WARNING", None)
        os.environ.pop("RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S", None)
        os.environ.pop("RAY_SCHEDULER_EVENTS", None)
        if isinstance(sys.stderr, StderrFilter):
            sys.stderr = sys.stderr.original_stderr


def init_ray(address: str = None, suppress_logging: bool = True, **kwargs):
    """
    Initialize Ray with optional warning suppression.

    When connecting via ray://, packages the project root as runtime_env
    so ray_hive is available on the cluster for serialization.
    """
    suppress_ray_warnings(suppress_logging)

    if address is None:
        address = os.getenv("RAY_ADDRESS")
        if not address:
            raise RuntimeError("RAY_ADDRESS not set. Copy .env.example to .env and set your cluster address.")

    if address.startswith("ray://") and "runtime_env" not in kwargs:
        kwargs["runtime_env"] = {"working_dir": _PROJECT_ROOT}

    ray.init(
        address=address,
        ignore_reinit_error=True,
        log_to_driver=not suppress_logging,
        configure_logging=not suppress_logging,
        **kwargs
    )


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


def is_node_alive(hostname: str) -> bool:
    """True when an Alive Ray node matches the registry hostname."""
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        if node.get("NodeManagerHostname") == hostname or node.get("NodeName") == hostname:
            return True
        resources = node.get("Resources") or {}
        if any(key.startswith(f"{hostname}_gpu") for key in resources):
            return True
    return False


def shutdown_all():
    """Shutdown all Serve apps and clear registry state."""
    from .deployment import get_deploy_service
    ray.get(get_deploy_service().shutdown_all.remote())


def shutdown_model(model_id: str):
    """Shutdown one model and clear its registry reservations."""
    from .deployment import get_deploy_service
    ray.get(get_deploy_service().shutdown_model.remote(model_id))


def kill_gpu_registry():
    """Kill detached singleton actors so they are recreated fresh on next init."""
    try:
        ray.kill(ray.get_actor("gpu_registry", namespace="system"))
    except ValueError:
        pass
    try:
        ray.kill(ray.get_actor("deploy_service", namespace="system"))
    except ValueError:
        pass
