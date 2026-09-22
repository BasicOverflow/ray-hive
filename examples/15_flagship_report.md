# Flagship deploy report — Qwen3.8-27B + Nemotron 3.5 Lightning

Date: 2026-09-07  
Harness: `examples/15_flagship_models.py`  
Cluster: `ray://10.0.1.52:10001`

## Sources

- [Qwen3.8-27B vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [Nemotron 3.5 Lightning day-0 vLLM blog](https://vllm.ai/blog/2026-08-10-nemotron-3-5-lightning-vllm)
- [Nemotron vLLM cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/vllm_cookbook.ipynb)
- GGUF sizing: [Unsloth Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) / community 3090 guidance (Q4_K_M on 24GB, smaller Q3 for context)

## Cluster GPUs (from Ray node resources)

| Host | Cards | Role |
|------|--------|------|
| **ergos-06-nv** | **1× RTX 3090 (24GB, Ampere sm86)** | Best single GPU. No native FP8. |
| **ergos-04-nv** | **2× RTX 4060 Ti (16GB + 8GB, Ada sm89)** | Only native-FP8 pair. Same-node TP=2; bottleneck is the 8GB card. |
| ergos-02-nv | 2× RTX 3060 Ti 8GB + 1× RTX 3060 12GB | Too small for 27B / 30B except last-resort TP. |

No H100 / GB200 / RTX 5090. Official datacenter recipes will not run as written.

## GGUF recommendations you asked to test

These are **llama.cpp / Unsloth Desktop** files, not vLLM checkpoints. Ray Hive cannot load `.gguf`.

| Quant | Typical size | Intent | vLLM stand-in on *this* cluster |
|-------|--------------|--------|----------------------------------|
| **Q4_K_M** | ~17 GB | Sweet spot for a **24GB 3090** | `cyankiwi/Qwen3.8-27B-AWQ-INT4` (W4A16, Marlin, Ampere-safe) pinned to `ergos-06-nv:gpu0` |
| **Q3_K_XL** | ~13 GB | More KV / **massive context** | Same W4A16, **TP=2 on the 4060 Ti pair** + `kv_cache_dtype=fp8` (Ada) and longer `max_input_prompt_length` |

## Ideal setups on *this* hardware

### Qwen3.8-27B

| Goal | Variant id | Checkpoint | Place | Spec decode |
|------|------------|------------|-------|-------------|
| Fastest decode that can actually start | `qwen-w4a16-3090` | W4A16 INT4 | 3090 TP=1 | **MTP**, 3 draft tokens (in-checkpoint) |
| Long context (Q3_K_XL analog) | `qwen-w4a16-longctx-4060` | W4A16 INT4 | 4060 Ti TP=2 | MTP + FP8 KV |
| Official recipe (likely OOM) | `qwen-fp8-4060` | `Qwen/Qwen3.8-27B-FP8` (~38GB) | 4060 Ti TP=2 | MTP; 19GB/GPU vs 16+8GB cards |

Recipe extras kept in EngineArgs: `reasoning_parser=qwen3`, `language_model_only`, `limit_mm_per_prompt`. Thinking off for bench (`enable_thinking: false`). Serve-only flags (`enable_auto_tool_choice`, `tool_call_parser`) must not be passed — this worker’s `AsyncEngineArgs` rejects them.

### Nemotron 3.5 Lightning (30B-A3B)

| Variant id | Why |
|------------|-----|
| `nemo-bf16-estimate` | Estimate only — BF16 ~60GB, no card here fits |

Nemotron BF16 on one 80GB H100 is the blog’s baseline. This cluster cannot host it without a much heavier quant that NVIDIA does not publish for Ampere.

## Live bench status

Last harness write is in `examples/15_flagship_report.json`. Qwen W4A16 planned (~21.3GB / 14 seqs on the 3090) then failed deploy on a leftover Serve CLI kwarg (`enable_auto_tool_choice`). That flag is stripped from the example now.

To finish the numbers:

1. Confirm `hive.get_vram_state()` lists the 3090 / 4060s.
2. `py examples/15_flagship_models.py`  
   Default `RUN` = `qwen-w4a16-3090` then `nemo-bf16-estimate`.
3. Results land in `examples/15_flagship_report.json` (req/s and approx tok/s on 32 short prompts).

First Qwen W4A16 pull is ~20+ GB. Budget time for HF download + CUDA compile on the worker.

## What “fastest speculative decoding” means here

- **Qwen:** MTP (`num_speculative_tokens: 3`) — no extra model. Acceptance is the thing to watch (`spec_decode_num_accepted` vs draft), not just tok/s.
- **Nemotron:** No deployable checkpoint on this cluster. The BF16 estimate is the documented path.
- Speculation **hurts** throughput when the GPU is already saturated (cookbook). This harness uses a 32-prompt batch after a 2-prompt warmup — treat tok/s as decode-ish, not ShareGPT-saturated.

## Recommendation (when the registry is healthy)

1. Run **`qwen-w4a16-3090`** first. That is the only variant that is both Ampere-legal and sized for the 3090, and it is the vLLM equivalent of the Q4_K_M advice.
2. If that is stable, run **`qwen-w4a16-longctx-4060`** for the Q3_K_XL / long-context experiment (FP8 KV needs Ada).
3. Keep **Nemotron** as `nemo-bf16-estimate` unless a smaller Ampere-safe checkpoint appears.
