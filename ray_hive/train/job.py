"""Ray Train job with optional VRAM registry claim."""
from __future__ import annotations

import logging
from typing import Any, Callable

from ray_hive.cluster.registry import ClaimKind, get_cluster_registry

logger = logging.getLogger("ray_hive.train.job")


class TrainJob:
    """Wrap TorchTrainer (or a callable) after claiming a train_run_id + optional GPUs.

    Does not replace vLLM ``deploy_model``. Shares ClusterRegistry + gpu_registry only.
    """

    def __init__(
        self,
        run_id: str,
        project_id: str,
        *,
        train_fn: Callable[..., Any] | None = None,
        num_workers: int = 1,
        claim_gpus: list[str] | None = None,
        use_gpu_registry: bool = True,
    ):
        self.run_id = run_id
        self.project_id = project_id
        self.train_fn = train_fn
        self.num_workers = num_workers
        self.claim_gpus = claim_gpus or []
        self.use_gpu_registry = use_gpu_registry
        self._claimed = False

    def _claim(self) -> None:
        import ray

        reg = get_cluster_registry()
        ray.get(
            reg.claim.remote(
                ClaimKind.TRAIN_RUN.value,
                self.run_id,
                self.project_id,
                meta={"num_workers": self.num_workers, "gpus": self.claim_gpus},
            )
        )
        self._claimed = True

    def _release(self) -> None:
        if not self._claimed:
            return
        import ray

        try:
            reg = get_cluster_registry()
            ray.get(reg.release.remote(ClaimKind.TRAIN_RUN.value, self.run_id, self.project_id))
        except Exception:
            pass
        self._claimed = False

    def run(self, *args, **kwargs) -> Any:
        self._claim()
        try:
            if self.use_gpu_registry and self.claim_gpus:
                try:
                    from ray_hive.core.gpu_registry import get_gpu_registry

                    # Soft check freest — detailed reserve left to caller / TorchTrainer
                    reg = get_gpu_registry()
                    logger.info("train job %s registry=%s", self.run_id, reg)
                except Exception as exc:
                    logger.warning("gpu_registry unavailable: %s", exc)
            if self.train_fn is None:
                raise RuntimeError("train_fn required (or pass TorchTrainer via train_fn)")
            # Prefer Ray Train when installed
            try:
                from ray.train.torch import TorchTrainer
                from ray.train import ScalingConfig

                if isinstance(self.train_fn, TorchTrainer):
                    return self.train_fn.fit()
                trainer = TorchTrainer(
                    train_loop_per_worker=self.train_fn,
                    scaling_config=ScalingConfig(
                        num_workers=self.num_workers,
                        use_gpu=bool(self.claim_gpus) or self.num_workers > 0,
                    ),
                    *args,
                    **kwargs,
                )
                return trainer.fit()
            except ImportError:
                return self.train_fn(*args, **kwargs)
        finally:
            self._release()
