"""Custom PyTorch / HF train + infer (optional ``ray-hive[train]``)."""
from .envelopes import gpu_remote
from .infer import CustomInferService
from .job import TrainJob

__all__ = ["gpu_remote", "TrainJob", "CustomInferService"]
