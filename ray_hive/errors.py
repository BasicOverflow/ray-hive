"""Ray Hive exception hierarchy."""


class RayHiveError(Exception):
    """Base for all Ray Hive errors."""


class ConfigError(RayHiveError, ValueError):
    """Bad user/config input."""


class InvalidGpuPinError(ConfigError):
    """gpu= pin shape / replicas mismatch."""


class ModelAlreadyDeployedError(ConfigError):
    """model_id already present in Serve or the GPU registry."""


class PlacementError(RayHiveError, ValueError):
    """GPU pin / auto-place / TP topology failed."""


class GpuNotFoundError(PlacementError):
    """Pinned GPU key not in the registry."""


class InsufficientVramError(PlacementError):
    """GPU free VRAM below required amount."""

    def __init__(self, message, *, gpu=None, available_gb=None, need_gb=None):
        super().__init__(message)
        self.gpu = gpu
        self.available_gb = available_gb
        self.need_gb = need_gb


class ArchRequirementError(PlacementError):
    """GPU compute capability / arch taint not met."""


class TpShardError(PlacementError):
    """Model dims not divisible for the requested TP size."""


class NoPlacementError(PlacementError):
    """Auto-place found no eligible GPU or TP group."""


class PlanningError(RayHiveError, ValueError):
    """VRAM / concurrency plan cannot be satisfied."""


class ModelDoesNotFitError(PlanningError):
    """Fixed non-KV / activations exceed the VRAM budget."""


class KvBudgetError(PlanningError):
    """No KV cache room after graph/sampler/util constraints."""


class MmContextError(PlanningError):
    """max_model_len cannot cover MM placeholders + output."""


class DeployError(RayHiveError, RuntimeError):
    """Serve deploy / replica bring-up failed."""


class InferenceError(RayHiveError, RuntimeError):
    """Runtime call path failed."""


class ModelNotFoundError(InferenceError):
    """Requested model_id is not a live Serve application."""

    def __init__(self, message, *, model_id=None):
        super().__init__(message)
        self.model_id = model_id


class UnsupportedModeError(InferenceError):
    """Operation not supported for this deploy mode or API."""


class MediaError(RayHiveError, ValueError):
    """Image/audio/video decode or content-part issues."""


def http_status_for(exc: BaseException) -> int:
    """Map a RayHiveError to an HTTP status code for Serve edges."""
    if isinstance(exc, ModelNotFoundError):
        return 404
    if isinstance(exc, (ConfigError, PlacementError, PlanningError, MediaError, UnsupportedModeError)):
        return 400
    if isinstance(exc, DeployError):
        return 502
    if isinstance(exc, RayHiveError):
        return 500
    raise TypeError(f"not a RayHiveError: {type(exc)!r}")
