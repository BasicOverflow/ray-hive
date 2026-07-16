"""Ray utility functions with warning suppression."""
import logging
import os
import re
import sys
import warnings

import ray
from dotenv import load_dotenv


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
