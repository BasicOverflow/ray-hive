"""Strip Qwen-style think blocks so Open WebUI does not abort the stream.

Qwen3.5/3.8 prefills ``<think>`` in the chat template. The model then emits a
short thought and ``</think>``. Open WebUI treats that close tag as end-of-reply
and drops everything after the first sentence.
"""

THINK_CLOSE = "</think>"
THINK_OPEN = "<think>"

# OpenWebUI / Qwen clients often send these as SamplingParams.stop, which
# cuts generation at the first line or at the think close (no answer).
_DROP_STOPS = frozenset({
    "\n",
    "\r",
    "\r\n",
    "\n\n",
    THINK_OPEN,
    THINK_CLOSE,
})


def strip_think(text: str) -> str:
    """Return text after the last ``</think>``, or the original if none."""
    if not text:
        return text
    idx = text.find(THINK_CLOSE)
    if idx == -1:
        return text[len(THINK_OPEN):] if text.startswith(THINK_OPEN) else text
    return text[idx + len(THINK_CLOSE):].lstrip("\n")


def sanitize_stop(stop):
    """Drop newline / think-tag stop strings that truncate a Qwen reply."""
    if stop is None:
        return None
    items = [stop] if isinstance(stop, str) else list(stop)
    cleaned = [s for s in items if s not in _DROP_STOPS]
    return cleaned or None


class ThinkStreamFilter:
    """Hold deltas until ``</think>`` is resolved so the close tag is never sent."""

    def __init__(self):
        self._buf = ""
        self._after_close = False

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        if self._after_close:
            return delta
        self._buf += delta
        idx = self._buf.find(THINK_CLOSE)
        if idx != -1:
            self._after_close = True
            before = self._buf[:idx]
            after = self._buf[idx + len(THINK_CLOSE):].lstrip("\n")
            self._buf = ""
            if before.startswith(THINK_OPEN):
                before = before[len(THINK_OPEN):]
            return before + after
        keep = len(THINK_CLOSE) - 1
        if len(self._buf) <= keep:
            return ""
        safe, self._buf = self._buf[:-keep], self._buf[-keep:]
        if safe.startswith(THINK_OPEN):
            safe = safe[len(THINK_OPEN):]
        return safe

    def flush(self) -> str:
        out = self._buf
        self._buf = ""
        if self._after_close:
            return out
        return strip_think(out)
