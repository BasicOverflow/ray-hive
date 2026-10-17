"""
RayHive client — high-level API for deploying models and querying VRAM state.

Wraps DeployService (serialized deploy) and gpu_registry (VRAM tracking).
"""
import time

import ray
from typing import Dict, List, Literal, Optional, Type, Union

from .core.deployment import get_deploy_service
from .core.gpu_alloc import BaseGpuAllocator
from .core.gpu_registry import get_gpu_registry
from .core.model_specs.attention import BaseAttentionSpecs
from .core.model_specs.estimate import load_hf_config_dict
from .core.model_specs.planner import normalize_hf_config
from .core.model_specs.vram_reqs import BaseVramReqs
from .core.ray_utils import init_ray, shutdown_all, shutdown_model, suppress_ray_warnings
from .core.ray_utils.display import (
    print_banner,
    print_deployment_plan,
    warn,
)
from .core.ray_utils.placement import plan_replica_groups, validate_auto_input_config
from .core.openai_gateway import ensure_openai_gateway
from .errors import ConfigError

# Lifted out of vllm_kwargs into planner config (must not also reach engine_kwargs).
_PLANNER_VLLM_KEYS = (
    "max_num_seqs",
    "max_num_batched_tokens",
    "auto_hybrid_input_cap",
    "sleep_peak_factor",
)


def _split_vllm_kwargs(vllm_kwargs: dict | None) -> tuple[dict, dict]:
    """Pop planner overrides from vllm_kwargs; return (overrides, remaining)."""
    remaining = dict(vllm_kwargs or {})
    overrides = {}
    for key in _PLANNER_VLLM_KEYS:
        if key in remaining:
            overrides[key] = remaining.pop(key)
    return overrides, remaining


def _registry_still_settling(exc: Exception) -> bool:
    """True when registry/DaemonSet has not finished reporting free VRAM yet."""
    msg = str(exc)
    return (
        "not in registry" in msg
        or "Known: []" in msg
        or "No GPUs in registry" in msg
        # Fresh registry entries start at free=0; nvidia-smi also reads 0 while
        # a just-killed engine is still releasing CUDA memory.
        or "0.00GB available" in msg
        or "0.0GB available" in msg
    )


def _retry_if_registry_empty(fn):
    """Poll briefly after kill_gpu_registry / teardown until DaemonSet VRAM is real."""
    last = None
    for attempt in range(8):
        try:
            return fn()
        except Exception as e:
            last = e
            if not _registry_still_settling(e) or attempt == 7:
                raise
            time.sleep(5)
    raise last


class RayHive:
    """Main client for distributed LLM serving."""

    def __init__(self, address: str, suppress_logging: bool = True, show_banner: bool = True, **kwargs):
        """Connect to Ray cluster at address and ensure singleton actors exist.

        Named hive actors live in Ray namespace ``ray_hive`` (override with
        ``namespace=`` or ``RAY_HIVE_NAMESPACE``). Extra kwargs go to ``ray.init``.
        """
        if show_banner:
            print_banner()
        warn(
            "WARNING: vLLM has model-family usage guides — check them before deploy "
            "so you pass the right HF config / vllm kwargs (quantization, rope, etc.). "
            "Those guides also recommend env vars per GPU architecture for better performance."
        )
        suppress_ray_warnings(suppress_logging)
        init_ray(address, suppress_logging=suppress_logging, **kwargs)
        get_gpu_registry()


    def deploy_model(
        self,
        model_id: str,
        model_name: str,
        max_input_prompt_length: Union[int, Literal["auto"]],
        max_output_prompt_length: int,
        replicas: int = 1,
        gpu: Optional[Union[str, List[str]]] = None,
        attention_cls: Optional[Type[BaseAttentionSpecs]] = None,
        vram_cls: Optional[Type[BaseVramReqs]] = None,
        allocation_cls: Optional[Type[BaseGpuAllocator]] = None,
        idle_timeout: int = -1,
        sleep_timeout: int = -1,
        vllm_kwargs: Optional[dict] = None,
    ) -> dict:
        """
        Deploy a model with VRAM-aware scheduling.

        Returns when the model is ready (router up + warmed). Status dict includes
        model_id, status="ready", route, and per-replica plan/GPU info.

        max_input_prompt_length and max_output_prompt_length are required.
        Pass planner overrides (max_num_seqs, max_num_batched_tokens,
        auto_hybrid_input_cap, sleep_peak_factor) inside vllm_kwargs — they are
        lifted automatically.
        sleep_peak_factor (default 0.0): when sleep_timeout>0, the registry keeps
        the running footprint reserved so a wake still fits. A positive factor
        also reserves that fraction of weights+draft. Do not set it unless a
        model really needs a second weight-sized peak.
        max_input_prompt_length=\"auto\" grows text input (floor 256) to fill VRAM at a
        fixed max_num_seqs (required in vllm_kwargs); output length stays fixed.
        When max_num_seqs==1, auto may shrink below 256 so weights + draft still fit.
        Hybrid models (GDN/Mamba layers without transformer KV) cap auto at 32768
        so leftover VRAM is not spent on an unbounded attention window. Raise
        auto_hybrid_input_cap to spend freed draft/KV room on a longer window.
        replicas=-1 deploys to all eligible GPUs (or all eligible TP groups when auto TP>1).
        attention_cls defaults to BaseAttentionSpecs (standard transformer KV sizing).
        vram_cls overrides the default VramReqs calculator (optional).
        allocation_cls defaults to RayPerformanceAllocator for single-GPU auto placement;
        ignored when gpu= is set. Auto TP packing always uses RayTensorParallelAllocator.
        gpu=None: place on one GPU if any fits; otherwise same-node TP (2, 3, ...).
        gpu=[a,b,...] + replicas=1: one same-node TP group.
        gpu=[a,b,...] + replicas=len(list): N single-GPU pins (one replica per GPU).
        sleep_timeout: seconds of inactivity before all replicas sleep (level 1); -1 never.
        idle_timeout: seconds of inactivity before full self-shutdown; -1 never.
        When both are set, idle_timeout must be greater than sleep_timeout.
        """
        if idle_timeout != -1 and idle_timeout <= 0:
            raise ConfigError("idle_timeout must be -1 (never) or a positive number of seconds")
        if sleep_timeout != -1 and sleep_timeout <= 0:
            raise ConfigError("sleep_timeout must be -1 (never) or a positive number of seconds")
        if idle_timeout > 0 and sleep_timeout > 0 and idle_timeout <= sleep_timeout:
            raise ConfigError("idle_timeout must be greater than sleep_timeout when both are set")
        planner_overrides, vllm_kwargs = _split_vllm_kwargs(vllm_kwargs)

        config = {
            "name": model_name,
            "replicas": replicas,
            "gpu": gpu,
            "max_input_prompt_length": max_input_prompt_length,
            "max_output_prompt_length": max_output_prompt_length,
            "attention_cls": attention_cls,
            "vram_cls": vram_cls,
            "allocation_cls": allocation_cls,
            "idle_timeout": idle_timeout,
            "sleep_timeout": sleep_timeout,
            **planner_overrides,
        }
        validate_auto_input_config(config)

        deploy_svc = get_deploy_service()

        def _deploy():
            return ray.get(deploy_svc.deploy_models.remote(
                model_configs={model_id: config},
                vllm_kwargs={model_id: vllm_kwargs},
            ))

        replicas_info = _retry_if_registry_empty(_deploy)[model_id]
        ensure_openai_gateway()
        # Sticky desired-state for supervisor revive (no-op unless HIVE_SUPERVISOR_STATE set).
        try:
            import os
            if os.environ.get("HIVE_SUPERVISOR_STATE"):
                from ray_hive.supervisor.desired_state import DesiredState, StickyModel

                DesiredState().pin(
                    StickyModel(
                        model_id=model_id,
                        model_name=model_name,
                        max_input_prompt_length=max_input_prompt_length,
                        max_output_prompt_length=max_output_prompt_length,
                        replicas=replicas,
                        gpu=gpu if isinstance(gpu, str) or gpu is None else list(gpu),
                        idle_timeout=idle_timeout,
                        sleep_timeout=sleep_timeout,
                        vllm_kwargs=vllm_kwargs,
                    )
                )
        except Exception:
            pass
        return {
            "model_id": model_id,
            "status": "ready",
            "route": f"/{model_id}",
            "openai_v1": "/v1",
            "replicas": replicas_info,
        }


    def ensure_openai_api(self, force: bool = False) -> str:
        """Start the cluster-wide OpenAI ``/v1`` gateway (idempotent). Returns route prefix."""
        from ray import serve

        serve.start(
            proxy_location="HeadOnly",
            http_options={"host": "0.0.0.0", "port": 8000},
        )
        ensure_openai_gateway(force=force)
        return "/v1"


    def estimate_vram(
        self,
        model_name: str,
        max_input_prompt_length: Union[int, Literal["auto"]],
        max_output_prompt_length: int,
        replicas: int = 1,
        gpu: Optional[Union[str, List[str]]] = None,
        attention_cls: Optional[Type[BaseAttentionSpecs]] = None,
        vram_cls: Optional[Type[BaseVramReqs]] = None,
        allocation_cls: Optional[Type[BaseGpuAllocator]] = None,
        vllm_kwargs: Optional[dict] = None,
    ) -> dict:
        """Dry-run packing against live GPUs; print the same plan deploy would use."""
        planner_overrides, vllm_kwargs = _split_vllm_kwargs(vllm_kwargs)
        vllm_kwargs.pop("default_chat_template_kwargs", None)
        vllm_kwargs.pop("chat_template", None)
        vllm_kwargs.pop("tensor_parallel_size", None)
        vllm_kwargs.pop("distributed_executor_backend", None)
        vllm_kwargs.pop("idle_timeout", None)
        vllm_kwargs.pop("sleep_timeout", None)

        config = {
            "name": model_name,
            "replicas": replicas,
            "gpu": gpu,
            "max_input_prompt_length": max_input_prompt_length,
            "max_output_prompt_length": max_output_prompt_length,
            "attention_cls": attention_cls,
            "vram_cls": vram_cls,
            "allocation_cls": allocation_cls,
            **planner_overrides,
        }
        validate_auto_input_config(config)

        hf_params = normalize_hf_config(load_hf_config_dict(model_name))

        def _plan():
            return plan_replica_groups(
                self.get_vram_state(), config, hf_params, vllm_kwargs, model_id=model_name
            )

        results = _retry_if_registry_empty(_plan)
        print_deployment_plan(model_name, results)
        return results


    def shutdown(self, model_id: Optional[str] = None):
        """Shutdown one model or all hive models (model_id=None). Never kills Serve globally."""
        if model_id is None:
            shutdown_all()
        else:
            shutdown_model(model_id)


    def get_vram_state(self, wait: bool = True, timeout_s: float = 40.0) -> Dict:
        """Return live VRAM state dict for all GPUs from the registry.

        When wait=True (default), poll until the DaemonSet has reported at least
        one GPU with free>0, or raise with an actionable timeout message.
        """
        registry = get_gpu_registry()
        if not wait:
            return ray.get(registry.get_all_gpus.remote())

        deadline = time.time() + timeout_s
        last: Dict = {}
        while True:
            last = ray.get(registry.get_all_gpus.remote())
            if last and any(float(g.get("free") or 0) > 0 for g in last.values()):
                return last
            if time.time() >= deadline:
                raise ConfigError(
                    "VRAM registry empty or still at free=0 after "
                    f"{timeout_s:.0f}s — check ray-vram-monitor DaemonSet, "
                    f"RAY_HIVE_NAMESPACE, and NODE_NAME on GPU workers. "
                    f"keys={list(last)}"
                )
            time.sleep(1.0)


    def free_gpus(self, min_free_gb: float = 4.0, *, unshared_only: bool = True) -> List[str]:
        """Alive GPUs with available >= min_free_gb, freest first (Lexis helper)."""
        from ray_hive.core.ray_utils.hardware import is_node_alive

        state = self.get_vram_state(wait=True)
        rows = []
        for key, gpu in state.items():
            host = key.split(":")[0]
            if not is_node_alive(host):
                continue
            avail = float(gpu.get("available") or 0)
            if avail < min_free_gb:
                continue
            pending = gpu.get("pending") or {}
            active = gpu.get("active") or {}
            if unshared_only and (pending or active):
                continue
            rows.append((avail, key))
        rows.sort(reverse=True)
        return [k for _, k in rows]
