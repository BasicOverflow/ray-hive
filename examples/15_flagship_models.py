"""Qwen3.8-27B + Nemotron 3.5 Lightning — cluster-aware deploy reference.

This cluster (see probe at top of run):
  ergos-06-nv  1x RTX 3090      24GB  Ampere (no native FP8)
  ergos-04-nv  2x RTX 4060 Ti   16GB + 8GB  Ada SM89 (FP8 ok; TP bottleneck is 8GB)
  ergos-02-nv  2x 3060 Ti 8GB + 1x 3060 12GB  Ampere (too small except last-resort TP)

Official recipes assume GB200 / H100 / 5090:
  https://recipes.vllm.ai/Qwen/Qwen3.8-27B
  https://vllm.ai/blog/2026-08-10-nemotron-3-5-lightning-vllm

GGUF notes you asked to test (llama.cpp, not vLLM):
  Q4_K_M  (~17GB)  — recommended for a 24GB 3090
  Q3_K_XL (~13GB)  — more KV room / massive context
vLLM equivalents on Ampere: W4A16 INT4 (Marlin), not GGUF.

Set RUN to the variant ids to estimate + deploy sequentially.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from ray_hive import RayHive
from ray_hive.core.model_specs import BaseAttentionSpecs
from ray_hive.core.ray_utils import approx_tdp, compute_cap, info, sm_count, success, warn

load_dotenv(Path(__file__).resolve().parent / ".env")

# --- hardware pins ---
GPU_3090 = "ergos-06-nv:gpu0"

# EngineArgs only — Serve CLI flags like enable_auto_tool_choice crash this vLLM.
QWEN_BASE = dict(
    trust_remote_code=True,
    reasoning_parser="qwen3",
    default_chat_template_kwargs={"enable_thinking": False},
    language_model_only=True,
    limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
    enable_prefix_caching=True,  # vLLM 0.28 default for hybrid/Mamba
    # Instruct-mode card defaults. DeepSeek Harness cannot send top_p / top_k /
    # presence_penalty, so bake them here or vLLM uses the thinking generation_config.
    override_generation_config={
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
)
NEMO_BASE = dict(
    trust_remote_code=True,
    reasoning_parser="nemotron_v3",
    default_chat_template_kwargs={"enable_thinking": False},
    enable_prefix_caching=True,
)

# syv-ai/qwen38-27b-rtx3090 starts here (true Marlin W4A16, ~19.5GB).
# Stock vLLM 0.28.0 has DFlash2 (#52816) but its fused-KV builder reads
# `qkv_proj.weight`, which packed W4A16 QKV does not have — so
# syvai/Qwen3.8-27B-DFlash2-W4A16 fails to load. Their 114–124 tok/s /
# 150k–262k numbers still need patches we do not have (int8 GEMMs, fp16
# DeltaNet, draft vocab, KVarN, DFLASH_TOKENS=15, quantized-QKV DFlash).
# Live path: AutoRound, no MTP, prefix cache, long ctx. UD-Q4_K_XL is llama.cpp.
# Dropped: official FP8 (~38GB) and W4A16 TP=2 on the 4060 Ti 16+8GB pair
# (8GB sibling cannot hold a shard).
QWEN_W4A16_AR = "dbirks/Qwen3.8-27B-W4A16-AutoRound"
QWEN_DFLASH2 = "syvai/Qwen3.8-27B-DFlash2-W4A16"
NEMO_BF16 = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


class Qwen38DraftAttention(BaseAttentionSpecs):
    """Qwen3.8 GQA; speculative draft weights/KV come from speculative_config."""

    @property
    def head_dim(self) -> int:
        if self.hf_params.get("head_dim") is not None:
            return int(self.hf_params["head_dim"])
        return super().head_dim

    @property
    def kv_heads(self) -> int:
        if self.hf_params.get("num_key_value_heads") is not None:
            return int(self.hf_params["num_key_value_heads"])
        return super().kv_heads


VARIANTS = {
    # syv-ai single-user rungs that exist on stock vLLM 0.28.0 (no patches).
    "qwen-dflash2-3090": {
        "description": "AutoRound + DFlash2-7 (stock 0.28 cannot load W4A16 QKV)",
        "keep": False,
        "config": {
            "model_name": QWEN_W4A16_AR,
            "max_input_prompt_length": "auto",
            "max_output_prompt_length": 8192,
            "replicas": 1,
            "gpu": GPU_3090,
            "attention_cls": Qwen38DraftAttention,
            "vllm_kwargs": {
                **QWEN_BASE,
                "dtype": "bfloat16",
                "max_num_seqs": 1,
                "max_num_batched_tokens": 1024,
                "speculative_config": {
                    "method": "dflash",
                    "model": QWEN_DFLASH2,
                    "num_speculative_tokens": 7,
                },
            },
        },
    },
    "qwen-autoround-mtp-3090": {
        "description": "AutoRound W4A16 + graphs + MTP-4, prefix cache, 8k out",
        "keep": True,
        "config": {
            "model_name": QWEN_W4A16_AR,
            "max_input_prompt_length": "auto",
            "max_output_prompt_length": 8192,
            "replicas": 1,
            "gpu": GPU_3090,
            "attention_cls": Qwen38DraftAttention,
            "vllm_kwargs": {
                **QWEN_BASE,
                "dtype": "bfloat16",
                "max_num_seqs": 1,
                "max_num_batched_tokens": 1024,
                "speculative_config": {"method": "mtp", "num_speculative_tokens": 4},
            },
        },
    },
    "qwen-autoround-3090": {
        "description": "AutoRound W4A16 + graphs, no MTP; extra VRAM → longer ctx",
        "keep": True,
        "config": {
            "model_name": QWEN_W4A16_AR,
            "max_input_prompt_length": "auto",
            "max_output_prompt_length": 8192,
            "replicas": 1,
            "gpu": GPU_3090,
            "attention_cls": Qwen38DraftAttention,
            "vllm_kwargs": {
                **QWEN_BASE,
                "dtype": "bfloat16",
                "max_num_seqs": 1,
                "max_num_batched_tokens": 1024,
                # Dropping MTP frees ~2.37 GiB. Default hybrid auto cap is 32k;
                # raise it so the planner spends that room on context.
                "auto_hybrid_input_cap": 131072,
            },
        },
    },
    "nemo-bf16-estimate": {
        "description": "Nemotron Lightning BF16 estimate only (~60GB; will not fit)",
        "estimate_only": True,
        "config": {
            "model_name": NEMO_BF16,
            "max_input_prompt_length": 2048,
            "max_output_prompt_length": 512,
            "replicas": 1,
            "vllm_kwargs": NEMO_BASE,
        },
    },
}

# MTP-4: ~2.37 GiB draft lm_head, hybrid auto stays at 32k.
RUN = [
    "qwen-autoround-mtp-3090",
]
FALLBACK = {
    "qwen-dflash2-3090": "qwen-autoround-mtp-3090",
    "qwen-autoround-mtp-3090": "qwen-autoround-3090",
}
# Free the 3090 before the swap.
SHUTDOWN_FIRST = [
    "qwen-dflash2-3090",
    "qwen-autoround-mtp-3090",
    "qwen-autoround-3090",
]

def print_cluster(hive: RayHive) -> dict:
    gpu_map = hive.get_vram_state()
    info(f"Cluster GPUs ({len(gpu_map)}):")
    for key, gpu in sorted(gpu_map.items()):
        specs = gpu.get("specs") or {}
        info(
            f"  {key}: {specs.get('name', '?')}  "
            f"avail={gpu.get('available', 0):.1f}/{gpu.get('total', 0):.1f}GB  "
            f"sm={sm_count(gpu)} cap={compute_cap(gpu)} tdp~{approx_tdp(gpu):.0f}W"
        )
    return gpu_map


def main():
    scheduler = RayHive(address=os.environ["RAY_ADDRESS"], suppress_logging=True)
    gpu_map = print_cluster(scheduler)
    if not gpu_map:
        warn("GPU registry empty — DaemonSet has not reported yet. Aborting deploys.")
        return

    for old_id in SHUTDOWN_FIRST:
        try:
            scheduler.shutdown(old_id)
            info(f"Shut down leftover {old_id}")
            time.sleep(15)
        except Exception as e:
            warn(f"shutdown {old_id}: {e}")

    queue = list(RUN)
    seen: set[str] = set()
    while queue:
        vid = queue.pop(0)
        if vid in seen:
            continue
        seen.add(vid)
        spec = VARIANTS[vid]
        cfg = spec["config"]
        deployed = False
        info(f"=== {vid}: {spec['description']} ===")
        try:
            scheduler.estimate_vram(**cfg)
        except Exception as e:
            warn(f"estimate failed: {e}")
            fb = FALLBACK.get(vid)
            if fb and fb not in seen:
                info(f"Falling back to {fb}")
                queue.append(fb)
            continue

        if spec.get("estimate_only"):
            continue

        try:
            status = scheduler.deploy_model(model_id=vid, **cfg)
            info(str(status))
            deployed = True
        except Exception as e:
            warn(f"deploy failed: {e}")
            fb = FALLBACK.get(vid)
            if fb and fb not in seen:
                info(f"Falling back to {fb}")
                queue.append(fb)
        finally:
            if spec.get("keep") and deployed:
                success(f"Leaving {vid} up")
            elif deployed:
                try:
                    scheduler.shutdown(vid)
                except Exception as e:
                    warn(f"shutdown {vid}: {e}")
                time.sleep(5)


if __name__ == "__main__":
    main()
