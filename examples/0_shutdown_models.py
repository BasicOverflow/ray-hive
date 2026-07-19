"""Shutdown Ray Serve deployments."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive, kill_gpu_registry

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

scheduler = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=True)
scheduler.shutdown()
kill_gpu_registry()
RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=True)