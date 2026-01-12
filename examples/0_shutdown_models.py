"""Shutdown Ray Serve deployments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vram_scheduler import VRAMScheduler

scheduler = VRAMScheduler(suppress_logging=False)
scheduler.shutdown()

# Shutdown specific model:
# scheduler.shutdown(model_id="qwen")
