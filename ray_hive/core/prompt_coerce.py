"""Prompt shaping for vLLM generate/encode."""


def coerce_engine_prompt(prompt, encode):
    """
    Send token ids for raw strings so vLLM skips the MM text renderer.

    Qwen3.5 VL + language_model_only still hits HfRenderer._mm_req_counter on
    string prompts (vLLM 0.25.x). Token-id prompts avoid that path.
    """
    if isinstance(prompt, str):
        return {"prompt_token_ids": encode(prompt)}
    return prompt
