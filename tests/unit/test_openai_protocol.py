"""OpenAI chat schema, tool parse, sampling clamp, model card."""
import json
from ray_hive.core.model_router import ChatMessage, ModelRouter
from ray_hive.core.openai_protocol import (
    ToolCallStreamFilter,
    clamp_max_tokens,
    flatten_tool_calls,
    hf_tool_calls,
    model_card,
    normalize_role,
    normalize_tools,
    parse_tool_calls,
    remap_tool_arguments,
)

_RouterCls = ModelRouter.func_or_class


def _bare_router(names=("a",), max_seqs=(1,), **meta_extra):
    r = object.__new__(_RouterCls)
    r.gpu_deployment_names = list(names)
    r.replica_metadata = {
        n: {"max_num_seqs": s, **meta_extra} for n, s in zip(names, max_seqs)
    }
    r._loads = {n: {"waiting": 0, "running": 0} for n in names}
    r._eng_start = 0
    r.model_id = "m"
    r.chat_template_kwargs = {}
    r.processor = None
    r.pooling = False
    r.multimodal = False
    return r


def test_chat_message_allows_null_content_and_tool_calls():
    msg = ChatMessage.model_validate({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "web_search", "arguments": "{}"},
        }],
    })
    assert msg.content is None
    assert msg.tool_calls[0].function.name == "web_search"


def test_chat_message_tool_result():
    msg = ChatMessage.model_validate({
        "role": "tool",
        "content": "ok",
        "tool_call_id": "call_1",
    })
    assert msg.tool_call_id == "call_1"
    assert msg.content == "ok"


def test_developer_maps_to_system():
    assert normalize_role("developer") == "system"
    assert normalize_role("user") == "user"
    r = _bare_router()
    hf, _ = r._parse_messages([
        ChatMessage.model_validate({"role": "developer", "content": "be brief"}),
        ChatMessage.model_validate({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "function": {"name": "x", "arguments": "{\"a\": 1}"},
            }],
        }),
        ChatMessage.model_validate({
            "role": "tool",
            "content": "1",
            "tool_call_id": "call_x",
        }),
    ])
    assert hf[0]["role"] == "system"
    assert hf[1]["tool_calls"][0]["function"]["name"] == "x"
    assert hf[2]["tool_call_id"] == "call_x"


def test_parse_hermes_tool_call():
    text = 'Sure.\n<tool_call>\n{"name": "web_search", "arguments": {"q": "Kiev"}}\n</tool_call>'
    content, calls = parse_tool_calls(text)
    assert content == "Sure."
    assert calls[0]["function"]["name"] == "web_search"
    assert '"q": "Kiev"' in calls[0]["function"]["arguments"] or '"q":"Kiev"' in calls[0]["function"]["arguments"]


def test_parse_dsh_xml_tool_call():
    text = (
        'I will search.\n'
        '<tool_call> <function=web_search> '
        '<parameter>queries> ["Byzantine Slavs"] </parameter> '
        '</function> </tool_call>'
    )
    content, calls = parse_tool_calls(text)
    assert content == "I will search."
    assert calls[0]["function"]["name"] == "web_search"
    args = calls[0]["function"]["arguments"]
    assert "Byzantine Slavs" in args


def test_parse_plain_text_no_tools():
    content, calls = parse_tool_calls("Just an essay.")
    assert content == "Just an essay."
    assert calls == []


def test_tool_stream_holds_opener():
    filt = ToolCallStreamFilter()
    out = filt.feed("I will look this up.\n<tool_call>")
    out += filt.feed('{"name": "web_search", "arguments": {}}</tool_call>')
    tail, calls = filt.flush()
    assert "I will look this up." in out
    assert "<tool_call>" not in out
    assert calls[0]["function"]["name"] == "web_search"
    assert tail == ""


def test_hf_tool_calls_keeps_openai_json_string():
    out = hf_tool_calls([{
        "id": "call_1",
        "type": "function",
        "function": {"name": "noop", "arguments": {"q": 1}},
    }])
    assert out[0]["function"]["arguments"] == '{"q": 1}'


def test_remap_tool_arguments_dict_and_string():
    msgs = [{
        "role": "assistant",
        "tool_calls": [{"function": {"name": "noop", "arguments": '{"q": 1}'}}],
    }]
    as_dict = remap_tool_arguments(msgs, as_dict=True)
    assert as_dict[0]["tool_calls"][0]["function"]["arguments"] == {"q": 1}
    as_str = remap_tool_arguments(as_dict, as_dict=False)
    assert json.loads(as_str[0]["tool_calls"][0]["function"]["arguments"]) == {"q": 1}
    flat = flatten_tool_calls(as_dict)
    assert flat[0]["tool_calls"][0]["name"] == "noop"


def test_apply_chat_template_retries_until_template_accepts():
    r = _bare_router()
    r.pooling = False
    r._template_style = None

    def apply(messages, **kwargs):
        args = messages[0]["tool_calls"][0]["function"]["arguments"]
        if not isinstance(args, dict):
            raise TypeError("Can only get item pairs from a mapping.")
        return "ok"

    r.tokenizer = type("T", (), {"apply_chat_template": staticmethod(apply)})()
    prompt = r._apply_chat_template(
        [{
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "noop", "arguments": "{}"}}],
        }],
        tools=[{"type": "function", "function": {"name": "noop"}}],
    )
    assert prompt == "ok"
    assert r._template_style[0] == "openai_dict"


def test_normalize_tools_and_legacy_functions():
    tools = normalize_tools(
        [{"type": "function", "function": {"name": "a", "parameters": {}}}],
        functions=[{"name": "b", "parameters": {}}],
    )
    assert [t["function"]["name"] for t in tools] == ["a", "b"]


def test_sampling_defaults_and_clamps_to_plan():
    r = _bare_router(max_output_prompt_length=64)
    try:
        defaulted = r._sampling_params(extra={"temperature": 0.0})
        clamped = r._sampling_params(max_tokens=4096, extra={"temperature": 0.0})
        small = r._sampling_params(max_tokens=8, extra={"temperature": 0.0})
    except Exception as e:
        import pytest
        pytest.skip(f"SamplingParams import/env: {e}")
    assert defaulted.max_tokens == 64
    assert clamped.max_tokens == 64
    assert small.max_tokens == 8


def test_clamp_helper_without_plan_is_passthrough():
    assert clamp_max_tokens(32, {"a": {"max_num_seqs": 1}}) == 32
    assert clamp_max_tokens(None, {"a": {"max_num_seqs": 1}}) is None


def test_model_card_includes_plan_caps():
    card = model_card(
        "m",
        {"rep": {"max_model_len": 8704, "max_output_prompt_length": 8192}},
        {"enable_thinking": False},
    )
    assert card["id"] == "m"
    assert card["context_window"] == 8704
    assert card["max_output_tokens"] == 8192
    assert card["reasoning"] is False


def test_is_openai_messages_without_content():
    r = _bare_router()
    assert r._is_openai_messages([{"role": "assistant", "tool_calls": []}])
