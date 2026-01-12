"""Ray utility functions with warning suppression."""
import ray
import os
import sys
import warnings
import logging


class StderrFilter:
    """Filters stderr to suppress Ray C++ warnings."""
    
    def __init__(self, original_stderr):
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
            "INFO 2026",
            "WARNING 2026",
            "[2026-",  # C++ warnings with timestamp
        ]
    
    def write(self, text):
        if not text.strip():
            self.original_stderr.write(text)
            return
        
        # Check if it's a C++ warning line (starts with [timestamp])
        if text.strip().startswith("[2026-") and (" W " in text or "INFO" in text):
            return
        
        # Check if it matches any suppress pattern
        if any(pattern in text for pattern in self.suppress_patterns):
            return
        
        self.original_stderr.write(text)
    
    def flush(self):
        self.original_stderr.flush()


def suppress_ray_warnings(suppress: bool = True):
    """Suppress all Ray warnings and logs."""
    if suppress:
        warnings.filterwarnings("ignore")
        logging.getLogger("ray").setLevel(logging.CRITICAL)
        logging.getLogger("ray.serve").setLevel(logging.CRITICAL)
        logging.getLogger("ray.util").setLevel(logging.CRITICAL)
        os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
        os.environ["RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S"] = "10"
        os.environ["RAY_SCHEDULER_EVENTS"] = "0"
        
        # Replace stderr with filtered version
        if not isinstance(sys.stderr, StderrFilter):
            _original_stderr = sys.stderr
            sys.stderr = StderrFilter(_original_stderr)
    else:
        # Restore logging if previously suppressed
        warnings.filterwarnings("default")
        logging.getLogger("ray").setLevel(logging.INFO)
        logging.getLogger("ray.serve").setLevel(logging.INFO)
        logging.getLogger("ray.util").setLevel(logging.INFO)
        os.environ.pop("RAY_DISABLE_IMPORT_WARNING", None)
        os.environ.pop("RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S", None)
        os.environ.pop("RAY_SCHEDULER_EVENTS", None)
        
        # Restore original stderr if it was filtered
        if isinstance(sys.stderr, StderrFilter):
            sys.stderr = sys.stderr.original_stderr


def init_ray(address: str = None, suppress_logging: bool = True, **kwargs):
    """Initialize Ray with optional warning suppression.
    
    Args:
        address: Ray cluster address
        suppress_logging: If True, suppress Ray warnings and logs. If False, show all logging.
        **kwargs: Additional arguments passed to ray.init()
    """
    suppress_ray_warnings(suppress_logging)
    
    if address is None:
        address = os.getenv("RAY_ADDRESS", "ray://10.0.1.53:10001")
    
    ray.init(
        address=address,
        ignore_reinit_error=True,
        log_to_driver=not suppress_logging,
        configure_logging=not suppress_logging,
        **kwargs
    )

