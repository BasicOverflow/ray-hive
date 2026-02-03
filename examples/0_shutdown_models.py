"""Shutdown Ray Serve deployments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ray_hive import RayHive

scheduler = RayHive(suppress_logging=True)
scheduler.shutdown()

