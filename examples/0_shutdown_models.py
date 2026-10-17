"""Shutdown hive Serve apps only (leaves Markets / other Serve apps alone)."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.core.ray_utils import success

load_dotenv(Path(__file__).resolve().parent / ".env")

scheduler = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=True, show_banner=False)
scheduler.shutdown()
success("All hive models shut down (gpu_registry kept for DaemonSet continuity).")
