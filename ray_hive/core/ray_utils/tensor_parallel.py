"""Tensor-parallel shardability checks."""


def assert_tp_shardable(hf_params: dict, tp_size: int) -> None:
    """vLLM TP needs heads/KV/vocab (padded to 64) divisible by tp_size."""
    if tp_size <= 1:
        return
    heads = hf_params.get("num_attention_heads") or hf_params.get("n_head")
    kv = hf_params.get("num_key_value_heads", heads)
    vocab = hf_params.get("vocab_size")
    if heads is not None and int(heads) % tp_size != 0:
        raise ValueError(
            f"num_attention_heads={heads} not divisible by tensor_parallel_size={tp_size}"
        )
    if kv is not None:
        kv = int(kv)
        if kv >= tp_size and kv % tp_size != 0:
            raise ValueError(
                f"num_key_value_heads={kv} not divisible by tensor_parallel_size={tp_size}"
            )
        if kv < tp_size and tp_size % kv != 0:
            raise ValueError(
                f"tensor_parallel_size={tp_size} not divisible by num_key_value_heads={kv}"
            )
    if vocab is not None:
        padded = ((int(vocab) + 63) // 64) * 64
        if padded % tp_size != 0:
            raise ValueError(
                f"vocab_size={vocab} (vLLM-padded {padded}) not divisible by "
                f"tensor_parallel_size={tp_size}"
            )


def tp_shardable(hf_params: dict, tp_size: int) -> bool:
    """True when model dims are compatible with this TP size."""
    try:
        assert_tp_shardable(hf_params, tp_size)
        return True
    except ValueError:
        return False
