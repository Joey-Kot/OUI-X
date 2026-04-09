from types import SimpleNamespace

from open_webui.utils.payload import (
    apply_model_params_as_defaults_openai,
    merge_model_params_with_base,
)


def test_model_params_do_not_override_explicit_request_params():
    model_params = {"temperature": 0.1}
    payload = {"temperature": 0.9}

    result = apply_model_params_as_defaults_openai(model_params, payload)

    assert result["temperature"] == 0.9


def test_model_params_apply_as_fallback_when_request_missing_value():
    model_params = {"top_p": 0.8}
    payload = {}

    result = apply_model_params_as_defaults_openai(model_params, payload)

    assert result["top_p"] == 0.8


def test_custom_params_are_fallback_and_do_not_override_request_values():
    model_params = {
        "temperature": 0.1,
        "custom_params": {"temperature": 0.2, "foo": "bar"},
    }
    payload = {"temperature": 0.9}

    result = apply_model_params_as_defaults_openai(model_params, payload)

    assert result["temperature"] == 0.9
    assert result["foo"] == "bar"


def test_model_system_prompt_is_prepended_when_request_already_has_system_message():
    model_params = {"system": "model system"}
    payload = {"messages": [{"role": "system", "content": "request system"}]}

    result = apply_model_params_as_defaults_openai(model_params, payload)

    assert result["messages"][0]["content"] == "model system\n\nrequest system"


def test_model_system_prompt_is_applied_when_request_has_no_system_message():
    model_params = {"system": "model system"}
    payload = {"messages": [{"role": "user", "content": "hi"}]}

    result = apply_model_params_as_defaults_openai(model_params, payload)

    assert result["messages"][0]["role"] == "system"
    assert result["messages"][0]["content"] == "model system"


def test_request_system_prompt_is_unchanged_when_model_has_no_system_prompt():
    model_params = {"temperature": 0.1}
    payload = {"messages": [{"role": "system", "content": "request system"}]}

    result = apply_model_params_as_defaults_openai(model_params, payload)

    assert result["messages"][0]["content"] == "request system"


def test_explicit_null_in_request_blocks_model_fallback_value():
    model_params = {"temperature": 0.1}
    payload = {"temperature": None}

    result = apply_model_params_as_defaults_openai(model_params, payload)

    assert "temperature" in result
    assert result["temperature"] is None


def test_model_default_custom_cache_params_apply_and_request_can_override():
    model_params = {
        "custom_params": {
            "prompt_cache_key": "model-default-key",
            "prompt_cache_retention": "24h",
        }
    }
    payload = {"prompt_cache_key": "request-key"}

    result = apply_model_params_as_defaults_openai(model_params, payload)

    assert result["prompt_cache_key"] == "request-key"
    assert result["prompt_cache_retention"] == "24h"


def test_merge_model_params_with_base_uses_custom_over_base():
    base_model = SimpleNamespace(
        params=SimpleNamespace(model_dump=lambda: {"temperature": 0.2, "top_p": 0.7})
    )
    custom_model = SimpleNamespace(
        base_model_id="base-id",
        params=SimpleNamespace(model_dump=lambda: {"temperature": 0.9}),
    )

    result = merge_model_params_with_base(
        model_info=custom_model,
        get_model_by_id=lambda _model_id: base_model,
    )

    assert result["temperature"] == 0.9
    assert result["top_p"] == 0.7


def test_request_params_override_custom_and_base_defaults():
    base_model = SimpleNamespace(
        params=SimpleNamespace(model_dump=lambda: {"temperature": 0.2, "top_p": 0.7})
    )
    custom_model = SimpleNamespace(
        base_model_id="base-id",
        params=SimpleNamespace(model_dump=lambda: {"temperature": 0.9}),
    )
    defaults = merge_model_params_with_base(
        model_info=custom_model,
        get_model_by_id=lambda _model_id: base_model,
    )

    result = apply_model_params_as_defaults_openai(defaults, {"temperature": 0.1})

    assert result["temperature"] == 0.1
    assert result["top_p"] == 0.7


def test_merge_model_params_with_base_recursively_merges_custom_params_json_strings():
    base_model = SimpleNamespace(
        params=SimpleNamespace(
            model_dump=lambda: {
                "custom_params": {
                    "response_format": '{"type":"json_schema","json_schema":{"name":"base","schema":{"type":"object","properties":{"a":{"type":"string"}}}}}'
                }
            }
        )
    )
    custom_model = SimpleNamespace(
        base_model_id="base-id",
        params=SimpleNamespace(
            model_dump=lambda: {
                "custom_params": {
                    "response_format": '{"json_schema":{"schema":{"properties":{"b":{"type":"number"}}}}}'
                }
            }
        ),
    )

    merged = merge_model_params_with_base(
        model_info=custom_model,
        get_model_by_id=lambda _model_id: base_model,
    )

    response_format = merged["custom_params"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "base"
    assert response_format["json_schema"]["schema"]["properties"]["a"]["type"] == "string"
    assert response_format["json_schema"]["schema"]["properties"]["b"]["type"] == "number"


def test_merge_model_params_with_base_keeps_non_json_custom_params_values():
    base_model = SimpleNamespace(
        params=SimpleNamespace(
            model_dump=lambda: {
                "custom_params": {"response_format": "not-json"}
            }
        )
    )
    custom_model = SimpleNamespace(
        base_model_id="base-id",
        params=SimpleNamespace(
            model_dump=lambda: {
                "custom_params": {"response_format": "still-not-json"}
            }
        ),
    )

    merged = merge_model_params_with_base(
        model_info=custom_model,
        get_model_by_id=lambda _model_id: base_model,
    )

    assert merged["custom_params"]["response_format"] == "still-not-json"
