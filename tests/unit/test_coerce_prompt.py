from ray_hive.core.prompt_coerce import coerce_engine_prompt


def test_coerce_string_to_token_ids():
    assert coerce_engine_prompt("hi", lambda s: [1, 2, 3]) == {"prompt_token_ids": [1, 2, 3]}


def test_coerce_leaves_dict():
    prompt = {"prompt_token_ids": [9]}
    assert coerce_engine_prompt(prompt, lambda s: [0]) is prompt
