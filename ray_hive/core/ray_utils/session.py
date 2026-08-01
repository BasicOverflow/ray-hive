"""Ray session helpers — connect, warnings, working_dir packaging."""
import logging
import os
import re
import sys
import warnings

import ray
import ray_hive
from ray_hive.errors import InferenceError


# Parent of the installed/editable `ray_hive` package — Ray Client working_dir.
_WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(ray_hive.__file__)))


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


def init_ray(address: str, suppress_logging: bool = True, **kwargs):
    """
    Initialize Ray with optional warning suppression.

    When connecting via ray://, packages the package parent as runtime_env
    working_dir so ray_hive is available on the cluster for serialization.
    """
    suppress_ray_warnings(suppress_logging)

    if address.startswith("ray://") and "runtime_env" not in kwargs:
        kwargs["runtime_env"] = {"working_dir": _WORKING_DIR}

    ray.init(
        address=address,
        ignore_reinit_error=True,
        log_to_driver=not suppress_logging,
        configure_logging=not suppress_logging,
        **kwargs
    )


def serve_base_url() -> str:
    """HTTP base for Ray Serve (``RAY_SERVE_URL`` or host from ``RAY_ADDRESS``)."""
    explicit = os.environ.get("RAY_SERVE_URL")
    if explicit:
        return explicit.rstrip("/")
    addr = os.environ.get("RAY_ADDRESS", "")
    if addr.startswith("ray://"):
        return f"http://{addr.removeprefix('ray://').split(':')[0]}:8000"
    if ray.is_initialized():
        return "http://127.0.0.1:8000"
    raise InferenceError("Set RAY_SERVE_URL or RAY_ADDRESS=ray://host:port for Serve HTTP")
