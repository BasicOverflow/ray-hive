"""OpenAI chat helpers — tool parse, role normalize, model card, stream hold."""
import json
import re
import uuid
from typing import Any


_HERMES_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_XML = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_PARAM_EQ = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)
_PARAM_GT = re.compile(r"<parameter>([^>]+)>(.*?)</parameter>", re.DOTALL)
_INVOKE = re.compile(
    r'<invoke\s+name="([^"]+)"[^>]*>(.*?)</invoke>',
    re.DOTALL | re.IGNORECASE,
)
_INVOKE_PARAM = re.compile(
    r'<parameter\s+name="([^"]+)"[^>]*>(.*?)</parameter>',
    re.DOTALL | re.IGNORECASE,
)
_TOOL_OPENERS = ("<tool_call", "<function=", "<invoke")


def normalize_role(role: str) -> str:
    """Map OpenAI ``developer`` onto ``system`` for chat templates."""
    return "system" if role == "developer" else role


def new_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _coerce_arg(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _tool_call(name: str, arguments: Any, call_id: str | None = None) -> dict:
    if isinstance(arguments, str):
        args = arguments
    else:
        args = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": call_id or new_tool_call_id(),
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def _args_from_xml_body(body: str) -> dict:
    args: dict[str, Any] = {}
    for m in _PARAM_EQ.finditer(body):
        args[m.group(1).strip()] = _coerce_arg(m.group(2))
    if args:
        return args
    for m in _PARAM_GT.finditer(body):
        args[m.group(1).strip()] = _coerce_arg(m.group(2))
    if args:
        return args
    for m in _INVOKE_PARAM.finditer(body):
        args[m.group(1).strip()] = _coerce_arg(m.group(2))
    return args


def _parse_hermes_obj(obj: dict) -> dict | None:
    name = obj.get("name") or (obj.get("function") or {}).get("name")
    if not name:
        return None
    arguments = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    return _tool_call(str(name), arguments, obj.get("id"))


def parse_tool_calls(text: str) -> tuple[str | None, list[dict]]:
    """Split assistant text into (content, OpenAI tool_calls).

    Accepts common dialects models emit: Hermes JSON in ``<tool_call>``,
    XML ``<function=name>``, and Anthropic-style ``<invoke name=...>``.
    """
    if not text:
        return (text or None), []

    calls: list[dict] = []
    spans: list[tuple[int, int]] = []

    for m in _HERMES_BLOCK.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        parsed = _parse_hermes_obj(obj)
        if parsed is None:
            continue
        calls.append(parsed)
        spans.append(m.span())

    covered = list(spans)
    for m in _TOOL_CALL_BLOCK.finditer(text):
        if any(start <= m.start() and m.end() <= end for start, end in covered):
            continue
        body = m.group(1)
        found = False
        for fm in _FUNCTION_XML.finditer(body):
            calls.append(_tool_call(fm.group(1).strip(), _args_from_xml_body(fm.group(2))))
            found = True
        if not found:
            for im in _INVOKE.finditer(body):
                calls.append(_tool_call(im.group(1).strip(), _args_from_xml_body(im.group(2))))
                found = True
        if found:
            spans.append(m.span())
            covered.append(m.span())

    for m in _FUNCTION_XML.finditer(text):
        if any(start <= m.start() and m.end() <= end for start, end in covered):
            continue
        calls.append(_tool_call(m.group(1).strip(), _args_from_xml_body(m.group(2))))
        spans.append(m.span())
        covered.append(m.span())

    for m in _INVOKE.finditer(text):
        if any(start <= m.start() and m.end() <= end for start, end in covered):
            continue
        calls.append(_tool_call(m.group(1).strip(), _args_from_xml_body(m.group(2))))
        spans.append(m.span())

    if not calls:
        return text, []

    content = text
    for start, end in sorted(spans, reverse=True):
        content = content[:start] + content[end:]
    content = content.strip()
    return (content or None), calls


def normalize_tools(tools: Any, functions: Any = None) -> list[dict] | None:
    """OpenAI tools list, including legacy ``functions``."""
    out: list[dict] = []
    if isinstance(tools, list):
        for item in tools:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function" or "function" in item:
                out.append(item)
            elif "name" in item:
                out.append({"type": "function", "function": item})
    if isinstance(functions, list):
        for item in functions:
            if isinstance(item, dict):
                out.append({"type": "function", "function": item})
    return out or None


def hf_tool_calls(tool_calls: list | None) -> list[dict] | None:
    """Canonical OpenAI tool_calls (``function.arguments`` is a JSON string)."""
    if not tool_calls:
        return None
    out = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            dump = tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
        else:
            dump = tc
        fn = dump.get("function") or {}
        args = fn.get("arguments", dump.get("arguments", {}))
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        out.append({
            "id": dump.get("id") or new_tool_call_id(),
            "type": dump.get("type") or "function",
            "function": {"name": fn.get("name") or dump.get("name") or "", "arguments": args},
        })
    return out


def _json_or_mapping(args: Any, as_dict: bool) -> Any:
    if as_dict:
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": args}
        return args if isinstance(args, dict) else {}
    if isinstance(args, str):
        return args
    return json.dumps(args, ensure_ascii=False)


def remap_tool_arguments(messages: list[dict], as_dict: bool) -> list[dict]:
    """Copy messages with tool ``arguments`` as dicts or JSON strings."""
    out = []
    for msg in messages:
        row = dict(msg)
        tcs = row.get("tool_calls")
        if not tcs:
            out.append(row)
            continue
        rewritten = []
        for tc in tcs:
            item = dict(tc)
            fn = dict(item.get("function") or {})
            args = _json_or_mapping(fn.get("arguments", item.get("arguments", {})), as_dict)
            if "function" in item or "name" not in item:
                fn["arguments"] = args
                if not fn.get("name") and item.get("name"):
                    fn["name"] = item["name"]
                item["function"] = fn
            else:
                item["arguments"] = args
            rewritten.append(item)
        row["tool_calls"] = rewritten
        out.append(row)
    return out


def flatten_tool_calls(messages: list[dict]) -> list[dict]:
    """``{name, arguments}`` tool_calls — some templates skip the function wrapper."""
    out = []
    for msg in messages:
        row = dict(msg)
        tcs = row.get("tool_calls")
        if not tcs:
            out.append(row)
            continue
        flat = []
        for tc in tcs:
            item = dict(tc)
            fn = item.get("function") or {}
            flat.append({
                "id": item.get("id"),
                "type": item.get("type") or "function",
                "name": fn.get("name") or item.get("name") or "",
                "arguments": fn.get("arguments", item.get("arguments", {})),
            })
        row["tool_calls"] = flat
        out.append(row)
    return out


def template_message_encodings(messages: list[dict]) -> list[tuple[str, list[dict]]]:
    """Encodings HF chat templates actually see in the wild.

    Order is cheapest-first: keep the OpenAI wire shape, then mapping args
    (Jinja ``.items()``), then flattened name/arguments.
    """
    as_str = remap_tool_arguments(messages, as_dict=False)
    as_dict = remap_tool_arguments(messages, as_dict=True)
    encodings = [
        ("openai_str", as_str),
        ("openai_dict", as_dict),
        ("flat_dict", flatten_tool_calls(as_dict)),
        ("flat_str", flatten_tool_calls(as_str)),
    ]
    seen: set[str] = set()
    unique = []
    for name, enc in encodings:
        key = json.dumps(enc, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, enc))
    return unique


def template_apply_errors() -> tuple:
    """Exceptions a chat template may raise for a bad tools/message encoding."""
    errs: list[type] = [TypeError, ValueError]
    try:
        from jinja2 import TemplateError
        errs.append(TemplateError)
    except ImportError:
        pass
    return tuple(errs)


def model_card(
    model_id: str,
    replica_metadata: dict | None = None,
    chat_template_kwargs: dict | None = None,
) -> dict:
    """OpenAI model object plus planned context / output caps."""
    card = {"id": model_id, "object": "model", "owned_by": "ray-hive"}
    meta = {}
    if replica_metadata:
        meta = next(iter(replica_metadata.values()), {}) or {}
    mml = meta.get("max_model_len")
    mout = meta.get("max_output_prompt_length")
    if mml:
        card["context_window"] = int(mml)
        card["max_model_len"] = int(mml)
    if mout is not None:
        card["max_output_tokens"] = int(mout)
        card["max_tokens"] = int(mout)
    thinking = bool((chat_template_kwargs or {}).get("enable_thinking"))
    card["reasoning"] = thinking
    return card


def planned_output_tokens(replica_metadata: dict | None) -> int | None:
    """Largest planned ``max_output_prompt_length`` across replicas."""
    if not replica_metadata:
        return None
    caps = []
    for meta in replica_metadata.values():
        v = (meta or {}).get("max_output_prompt_length")
        if v:
            caps.append(int(v))
    return max(caps) if caps else None


def clamp_max_tokens(requested: int | None, replica_metadata: dict | None) -> int | None:
    """Default / clamp to the planned output length when one is known."""
    cap = planned_output_tokens(replica_metadata)
    if cap is None:
        return requested
    if requested is None:
        return cap
    return max(1, min(int(requested), cap))


def finish_reason_for(output, tool_calls: list | None, max_tokens: int | None) -> str:
    """Prefer tool_calls, else vLLM finish_reason, else stop."""
    if tool_calls:
        return "tool_calls"
    fr = None
    outs = getattr(output, "outputs", None) if output is not None else None
    if outs:
        fr = getattr(outs[0], "finish_reason", None)
    if fr == "length":
        return "length"
    if fr:
        return str(fr)
    if max_tokens and output is not None and outs:
        n = len(getattr(outs[0], "token_ids", None) or [])
        if n >= int(max_tokens):
            return "length"
    return "stop"


class ToolCallStreamFilter:
    """Pass text until a tool-call opener, then hold the rest for parse."""

    def __init__(self):
        self._buf = ""
        self._holding = False
        self._hold = ""
        self._max_prefix = max(len(o) for o in _TOOL_OPENERS)

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        if self._holding:
            self._hold += delta
            return ""
        self._buf += delta
        for i, ch in enumerate(self._buf):
            if ch != "<":
                continue
            rest = self._buf[i:]
            if any(rest.startswith(o) for o in _TOOL_OPENERS):
                safe, self._hold = self._buf[:i], self._buf[i:]
                self._buf = ""
                self._holding = True
                return safe
        if len(self._buf) > self._max_prefix:
            safe, self._buf = self._buf[:-self._max_prefix], self._buf[-self._max_prefix:]
            return safe
        return ""

    def flush(self) -> tuple[str, list[dict]]:
        raw = self._hold if self._holding else self._buf
        self._buf = ""
        self._hold = ""
        self._holding = False
        if not raw:
            return "", []
        content, calls = parse_tool_calls(raw)
        if calls:
            return (content or ""), calls
        return raw, []
