"""
GPU registry: deployment reservations + live VRAM state from cluster daemon.

ClusterStateManager tracks logical deployment VRAM reservations.
VRAMAllocator adds live per-GPU state updated by the cluster DaemonSet.
RayGpuRegistry wraps VRAMAllocator as a detached Ray actor (name: gpu_registry).
"""
from abc import ABC, abstractmethod

import ray

from ray_hive.errors import InsufficientVramError


class ClusterStateManager(ABC):
    """Tracks router-level deployments and replica GPU reservations."""

    def __init__(self):
        """Initialize empty deployment registry."""
        self.deployments = {}


    @abstractmethod
    def get_gpu_vram(self, gpu_key: str) -> dict | None:
        """Return live GPU state for one GPU."""


    @abstractmethod
    def get_all_gpus(self) -> dict[str, dict]:
        """Return live GPU state for all GPUs."""


    def _live_free_vram_gb(self) -> dict[str, float]:
        """Return per-GPU available VRAM from live registry state."""
        return {gpu_key: gpu["available"] for gpu_key, gpu in self.get_all_gpus().items()}


    def _pending_covers_request(self, replica_gpu_vram_gb: dict[str, dict[str, float]]) -> bool:
        """True when every replica is already held in pending with enough VRAM."""
        for replica_id, gpu_vram_gb in replica_gpu_vram_gb.items():
            for gpu_key, vram_gb in gpu_vram_gb.items():
                gpu = self.get_gpu_vram(gpu_key)
                if gpu is None or gpu["pending"].get(replica_id, 0.0) < vram_gb:
                    return False
        return True


    def _check_capacity(self, replica_gpu_vram_gb: dict[str, dict[str, float]], deployment_id: str):
        """Raise if requested replica VRAM exceeds available capacity on any GPU."""
        if self._pending_covers_request(replica_gpu_vram_gb):
            return

        requested_by_gpu = {}
        for gpu_vram_gb in replica_gpu_vram_gb.values():
            for gpu_key, vram_gb in gpu_vram_gb.items():
                requested_by_gpu[gpu_key] = requested_by_gpu.get(gpu_key, 0.0) + vram_gb

        live_free = self._live_free_vram_gb()
        for gpu_key, requested_gb in requested_by_gpu.items():
            # available is free - pending; do not subtract used_vram again
            available_gb = live_free.get(gpu_key, 0.0)
            if available_gb < requested_gb:
                raise InsufficientVramError(
                    f"Not enough VRAM on {gpu_key} for deployment {deployment_id}: "
                    f"requested {requested_gb} GiB, available {available_gb} GiB",
                    gpu=gpu_key,
                    available_gb=available_gb,
                    need_gb=requested_gb,
                )


    def reserve_deployment(
        self,
        deployment_id: str,
        replica_gpu_vram_gb: dict[str, dict[str, float]],
        deployment_type: str = "replica",
        model_id: str | None = None,
    ):
        """Register replica VRAM reservations for a deployment after capacity check."""
        deployment = self.deployments.get(deployment_id)

        for replica_id in replica_gpu_vram_gb:
            assert not (deployment and replica_id in deployment["replicas"]), (
                f"Replica {replica_id} already exists for deployment {deployment_id}"
            )

        self._check_capacity(replica_gpu_vram_gb, deployment_id)

        if deployment is None:
            deployment = {"deployment_id": deployment_id, "deployment_type": deployment_type, "replicas": {}, "model_id": model_id}
            self.deployments[deployment_id] = deployment

        for replica_id, gpu_vram_gb in replica_gpu_vram_gb.items():
            deployment["replicas"][replica_id] = {"replica_id": replica_id, "gpu_vram_gb": gpu_vram_gb}

        if model_id is not None:
            deployment["model_id"] = model_id


    def release_deployment(self, deployment_id: str):
        """Remove a deployment and all its replica reservations."""
        self.deployments.pop(deployment_id, None)


    def has_deployment(self, deployment_id: str) -> bool:
        """True when deployment_id is already registered."""
        return deployment_id in self.deployments


    def get_deployment(self, deployment_id: str) -> dict | None:
        """Return deployment record or None."""
        return self.deployments.get(deployment_id)


    def list_model_ids(self) -> list[str]:
        """Return model_ids for deployments registered as type ``model``."""
        out = []
        for deployment_id, dep in self.deployments.items():
            if dep.get("deployment_type") != "model":
                continue
            out.append(dep.get("model_id") or deployment_id)
        return out


class VRAMAllocator(ClusterStateManager):
    """In-memory live GPU state updated by cluster daemon."""

    def __init__(self):
        """Initialize empty GPU state and deployment registry."""
        super().__init__()
        self.gpus = {}


    def _gpu_key(self, node_id: str, gpu_id: int) -> str:
        """Return canonical GPU key: k8s_hostname:gpuN (never Ray hex node ids)."""
        # Legacy monitors briefly used Ray's hex node id — reject so they can't reappear.
        assert not (len(node_id) >= 32 and all(c in "0123456789abcdef" for c in node_id.lower())), (
            f"node_id must be k8s hostname from NODE_NAME, got Ray node id {node_id!r}"
        )
        return f"{node_id}:gpu{gpu_id}"


    def _new_gpu_entry(self) -> dict:
        """Return a fresh per-GPU state dict."""
        return {
            "total": 0.0,
            "free": 0.0,
            "pending": {},
            "active": {},
            "specs": {},
        }


    def _gpu_view(self, gpu_key: str) -> dict | None:
        """Return public view of one GPU — all stored fields plus computed available VRAM."""
        if gpu_key not in self.gpus:
            return None
        gpu = self.gpus[gpu_key]
        view = dict(gpu)
        view["available"] = gpu["free"] - sum(gpu["pending"].values())
        return view


    def update_gpu(self, node_id: str, gpu_id: int, free_gb: float, total_gb: float):
        """Update live VRAM totals from nvidia-smi (called by DaemonSet)."""
        gpu_key = self._gpu_key(node_id, gpu_id)
        if gpu_key not in self.gpus:
            self.gpus[gpu_key] = self._new_gpu_entry()
        self.gpus[gpu_key]["total"] = total_gb
        self.gpus[gpu_key]["free"] = free_gb


    def update_gpu_specs(self, node_id: str, gpu_id: int, pycuda_specs: dict):
        """Store raw PyCUDA device_attribute dump for a GPU."""
        gpu_key = self._gpu_key(node_id, gpu_id)
        if gpu_key not in self.gpus:
            self.gpus[gpu_key] = self._new_gpu_entry()
        self.gpus[gpu_key]["specs"] = pycuda_specs


    def get_available_vram(self, gpu_key: str) -> float:
        """Return available VRAM on a GPU: free minus pending reservations."""
        view = self._gpu_view(gpu_key)
        return view["available"] if view else 0.0


    def reserve_replica(self, replica_id: str, gpu_key: str, vram_gb: float) -> bool:
        """Create a pending VRAM reservation for a replica before model load."""
        if gpu_key not in self.gpus:
            return False
        if self.get_available_vram(gpu_key) < vram_gb:
            return False
        self.gpus[gpu_key]["pending"][replica_id] = vram_gb
        return True


    def mark_initialized(self, replica_id: str, gpu_key: str):
        """Move a replica reservation from pending to active after successful init."""
        if gpu_key not in self.gpus:
            return
        gpu = self.gpus[gpu_key]
        if replica_id in gpu["pending"]:
            gpu["active"][replica_id] = gpu["pending"].pop(replica_id)


    def get_gpu_vram(self, gpu_key: str) -> dict | None:
        """Return public VRAM view for one GPU."""
        return self._gpu_view(gpu_key)


    def get_all_gpus(self) -> dict[str, dict]:
        """Return public VRAM views for all known GPUs."""
        return {gpu_key: self._gpu_view(gpu_key) for gpu_key in self.gpus}


    def clear_all(self) -> int:
        """Clear all pending/active reservations and deployment registry."""
        cleared = 0
        for gpu in self.gpus.values():
            cleared += len(gpu["pending"]) + len(gpu["active"])
            gpu["pending"] = {}
            gpu["active"] = {}
        self.deployments = {}
        return cleared


    def clear_replicas(self, replica_ids: list[str]) -> int:
        """Clear pending/active reservations for exact replica ids only."""
        ids = set(replica_ids)
        cleared = 0
        for gpu in self.gpus.values():
            for rid in list(gpu["pending"]):
                if rid in ids:
                    gpu["pending"].pop(rid)
                    cleared += 1
            for rid in list(gpu["active"]):
                if rid in ids:
                    gpu["active"].pop(rid)
                    cleared += 1
        return cleared


@ray.remote(num_cpus=0)
class RayGpuRegistry(VRAMAllocator):
    """Ray actor wrapper for VRAMAllocator (detached singleton)."""
    pass


def get_gpu_registry():
    """Get or create the detached gpu registry actor."""
    try:
        return ray.get_actor("gpu_registry", namespace="system")
    except ValueError:
        return RayGpuRegistry.options(
            name="gpu_registry",
            namespace="system",
            lifetime="detached",
        ).remote()
