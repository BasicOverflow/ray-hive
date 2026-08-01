"""N — error HTTP map (actor paths covered lightly)."""
import pytest

from ray_hive.errors import (
    ArchRequirementError,
    ConfigError,
    DeployError,
    GpuNotFoundError,
    InferenceError,
    InsufficientVramError,
    InvalidGpuPinError,
    KvBudgetError,
    MediaError,
    MmContextError,
    ModelAlreadyDeployedError,
    ModelDoesNotFitError,
    ModelNotFoundError,
    NoPlacementError,
    PlacementError,
    PlanningError,
    RayHiveError,
    TpShardError,
    UnsupportedModeError,
    http_status_for,
)


@pytest.mark.parametrize(
    "exc,code",
    [
        (ModelNotFoundError("x"), 404),
        (ConfigError("x"), 400),
        (InvalidGpuPinError("x"), 400),
        (ModelAlreadyDeployedError("x"), 400),
        (PlacementError("x"), 400),
        (GpuNotFoundError("x"), 400),
        (InsufficientVramError("x"), 400),
        (ArchRequirementError("x"), 400),
        (TpShardError("x"), 400),
        (NoPlacementError("x"), 400),
        (PlanningError("x"), 400),
        (ModelDoesNotFitError("x"), 400),
        (KvBudgetError("x"), 400),
        (MmContextError("x"), 400),
        (MediaError("x"), 400),
        (UnsupportedModeError("x"), 400),
        (InferenceError("x"), 500),
        (DeployError("x"), 502),
        (RayHiveError("x"), 500),
    ],
)
def test_http_status_leaves(exc, code):
    assert http_status_for(exc) == code
