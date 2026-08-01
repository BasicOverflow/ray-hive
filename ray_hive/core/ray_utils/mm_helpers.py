"""Shared helpers for multimodal token / limit parsing."""


def mm_count(limit_mm_per_prompt: dict | None, modality: str) -> int:
    """Return max items for a modality from limit_mm_per_prompt."""
    if not limit_mm_per_prompt:
        return 0
    val = limit_mm_per_prompt.get(modality, 0)
    if isinstance(val, dict):
        return int(val.get("count", 0) or 0)
    return int(val or 0)


def mm_size_hint(limit_mm_per_prompt: dict | None, modality: str, key: str, default: int) -> int:
    """Return optional size hint (width/height/num_frames/length) from limit entry."""
    if not limit_mm_per_prompt:
        return default
    val = limit_mm_per_prompt.get(modality)
    if isinstance(val, dict) and val.get(key) is not None:
        return int(val[key])
    return default


def vision_tokens_per_image(
    hf_params: dict,
    limit_mm_per_prompt: dict | None = None,
    mm_processor_kwargs: dict | None = None,
) -> int:
    """
    Worst-case placeholder tokens for one image.

    Prefers Gemma-style soft-token budgets (mm_processor_kwargs.max_soft_tokens
    or vision_soft_tokens_per_image). Else uses vision_config patch grid.
    """
    proc = mm_processor_kwargs or {}
    vision = hf_params.get("vision_config")
    if not isinstance(vision, dict):
        vision = {}

    soft = proc.get("max_soft_tokens")
    if soft is None:
        soft = hf_params.get("vision_soft_tokens_per_image")
    if soft is None:
        soft = vision.get("num_soft_tokens") or vision.get("default_output_length")
    if soft is not None:
        return max(1, int(soft))

    patch = int(vision.get("patch_size") or hf_params.get("patch_size") or 14)
    merge = int(
        vision.get("spatial_merge_size")
        or hf_params.get("spatial_merge_size")
        or 1
    )
    merge = max(1, merge)

    width = mm_size_hint(limit_mm_per_prompt, "image", "width", 0)
    height = mm_size_hint(limit_mm_per_prompt, "image", "height", 0)
    if width > 0 and height > 0:
        tokens = (width // patch) * (height // patch) // (merge * merge)
        return max(1, tokens)

    image_size = int(vision.get("image_size") or hf_params.get("image_size") or 448)
    tokens = (image_size // patch) ** 2 // (merge * merge)
    # Qwen2-VL-style dynamic res can be larger than square image_size; pad up.
    return max(256, tokens)


# Gemma4 / prefix-LM video: lower soft-token budget per frame than images.
_VIDEO_SOFT_TOKENS = 70
_VIDEO_BOUNDARY_TOKENS = 2 + 6  # boi/eoi + timestamps
_VIDEO_MAX_FRAMES = 32


def vision_tokens_per_video(
    hf_params: dict,
    limit_mm_per_prompt: dict | None = None,
    mm_processor_kwargs: dict | None = None,
) -> int:
    """Worst-case placeholder tokens for one video (frames × per-frame budget)."""
    frames = mm_size_hint(limit_mm_per_prompt, "video", "num_frames", _VIDEO_MAX_FRAMES)
    frames = max(1, min(frames, _VIDEO_MAX_FRAMES))
    # Explicit WxH grid overrides the soft-token video path.
    vision = hf_params.get("vision_config")
    if not isinstance(vision, dict):
        vision = {}
    width = mm_size_hint(limit_mm_per_prompt, "video", "width", 0)
    height = mm_size_hint(limit_mm_per_prompt, "video", "height", 0)
    if width > 0 and height > 0:
        patch = int(vision.get("patch_size") or hf_params.get("patch_size") or 14)
        merge = max(1, int(vision.get("spatial_merge_size") or hf_params.get("spatial_merge_size") or 1))
        per_frame = max(1, (width // patch) * (height // patch) // (merge * merge))
        return frames * per_frame
    return frames * (_VIDEO_SOFT_TOKENS + _VIDEO_BOUNDARY_TOKENS)


def audio_tokens_per_item(hf_params: dict, limit_mm_per_prompt: dict | None = None) -> int:
    """Worst-case placeholder tokens for one audio clip."""
    length = mm_size_hint(limit_mm_per_prompt, "audio", "length", 0)
    audio = hf_params.get("audio_config")
    if not isinstance(audio, dict):
        audio = {}
    # Coarse: ~50 tokens/sec of audio at 16kHz with typical hop; length hint in samples.
    if length > 0:
        hop = int(audio.get("hop_length") or 160)
        return max(1, length // max(hop, 1))
    return int(audio.get("max_source_positions") or hf_params.get("max_audio_tokens") or 750)


def estimate_encoder_params(config: dict, default_hidden: int = 1024) -> float:
    """Coarse parameter count for a vision/audio encoder tower from its config dict."""
    hidden = int(
        config.get("hidden_size")
        or config.get("width")
        or config.get("d_model")
        or default_hidden
    )
    layers = int(
        config.get("num_hidden_layers")
        or config.get("depth")
        or config.get("encoder_layers")
        or 24
    )
    intermediate = int(
        config.get("intermediate_size")
        or config.get("encoder_ffn_dim")
        or hidden * 4
    )
    n_heads = int(config.get("num_attention_heads") or config.get("num_heads") or 0)
    head_dim = int(config.get("head_dim") or (hidden // n_heads if n_heads else hidden))
    kv_heads = int(config.get("num_key_value_heads") or n_heads or 1)
    # Q+O use q width; K+V use kv width (GQA-aware).
    q_width = (n_heads * head_dim) if n_heads else hidden
    kv_width = kv_heads * head_dim
    attn = 2 * hidden * q_width + 2 * hidden * kv_width
    act = str(
        config.get("hidden_act")
        or config.get("hidden_activation")
        or config.get("mlp_hidden_act")
        or config.get("activation_function")
        or "gelu"
    ).lower()
    if act in ("gelu", "gelu_new", "gelu_pytorch_tanh", "gelu_fast", "relu", "relu2", "tanh"):
        mlp = 2 * hidden * intermediate
    else:
        # silu / swiglu: gate + up + down
        mlp = 3 * hidden * intermediate
    patch = int(config.get("patch_size") or 0)
    temporal = int(config.get("temporal_patch_size") or 1)
    in_chans = int(config.get("in_chans") or config.get("num_channels") or (3 if patch else 0))
    patch_embed = 0.0
    if in_chans > 0 and patch > 0:
        patch_embed = float(in_chans * temporal * patch * patch * hidden)
    # Absolute / learned position tables (Gemma4 vision: position_embedding_size).
    pos = int(config.get("position_embedding_size") or config.get("num_positions") or 0)
    pos_embed = float(pos * hidden) if pos > 0 else 0.0
    # Audio / tower output projection into LM space.
    out_dims = int(config.get("output_proj_dims") or config.get("out_hidden_size") or 0)
    out_proj = float(hidden * out_dims) if out_dims > 0 else 0.0
    return float(layers * (attn + mlp) + patch_embed + pos_embed + out_proj)
