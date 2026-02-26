from open_webui.utils.payload import apply_model_params_as_defaults_openai


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
