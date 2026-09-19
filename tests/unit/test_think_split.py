from ray_hive.core.think_split import ThinkStreamFilter, sanitize_stop, strip_think


def test_strip_think_drops_prefill_close_tag():
    assert strip_think("Paris is the capital.\n</think>\n\nThe answer is Paris.") == "The answer is Paris."
    assert strip_think("Just the answer.") == "Just the answer."
    assert strip_think("<think>hidden") == "hidden"


def test_sanitize_stop_drops_newline_and_think_tags():
    assert sanitize_stop(["\n", "</think>", "<|im_end|>"]) == ["<|im_end|>"]
    assert sanitize_stop(["\n", "</think>"]) is None
    assert sanitize_stop("\n") is None


def test_stream_filter_never_emits_close_tag():
    filt = ThinkStreamFilter()
    out = []
    for piece in ("The sky is blue.\n", "</think>", "\n\nIt is a gas giant."):
        out.append(filt.feed(piece))
    out.append(filt.flush())
    joined = "".join(out)
    assert "</think>" not in joined
    assert "The sky is blue." in joined
    assert "It is a gas giant." in joined


def test_stream_filter_split_close_tag_across_chunks():
    filt = ThinkStreamFilter()
    out = filt.feed("Hello.</thi") + filt.feed("nk>\n\nWorld") + filt.flush()
    assert "Hello" in out and "World" in out
    assert "</think>" not in out


def test_stream_filter_no_tag_passthrough():
    filt = ThinkStreamFilter()
    out = filt.feed("Hello world.") + filt.flush()
    assert out == "Hello world."
