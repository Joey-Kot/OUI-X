from types import SimpleNamespace

from open_webui.utils import completion_adapter as adapter


def _user(user_id: str = "user-1", role: str = "user"):
    return SimpleNamespace(id=user_id, role=role)


def test_resolve_endpoint_kind_prefers_explicit_value():
    assert (
        adapter.resolve_endpoint_kind(
            explicit_endpoint="responses", provider_type="openai"
        )
        == "responses"
    )
    assert (
        adapter.resolve_endpoint_kind(
            explicit_endpoint="chat_completions", provider_type="openai_responses"
        )
        == "chat_completions"
    )


def test_resolve_endpoint_kind_falls_back_to_provider_type():
    assert adapter.resolve_endpoint_kind(provider_type="openai_responses") == "responses"
    assert adapter.resolve_endpoint_kind(provider_type="openai") == "chat_completions"


def test_provider_type_from_model_id_prefers_openai_models():
    models = {"m-1": {"provider_type": "openai"}}
    openai_models = {"m-1": {"provider_type": "openai_responses"}}
    assert (
        adapter.provider_type_from_model_id(
            model_id="m-1", models=models, openai_models=openai_models
        )
        == "openai_responses"
    )


def test_normalize_tools_for_responses_converts_chat_shape():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather_lookup",
                "description": "Lookup weather",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            },
        }
    ]

    result = adapter.normalize_tools_for_responses(tools)
    assert result[0]["type"] == "function"
    assert result[0]["name"] == "weather_lookup"
    assert "function" not in result[0]
    assert result[0]["strict"] is True


def test_chat_messages_to_responses_input_handles_tool_calls_and_outputs():
    messages = [
        {
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "weather_lookup", "arguments": {"city": "Paris"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": {"ok": True}},
    ]

    result = adapter.chat_messages_to_responses_input(messages)
    assert result[0]["type"] == "function_call"
    assert result[0]["name"] == "weather_lookup"
    assert result[1]["role"] == "assistant"
    assert result[2]["type"] == "function_call_output"
    assert result[2]["call_id"] == "call-1"


def test_build_upstream_payload_for_responses_strips_internal_keys():
    form_data = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "foo", "parameters": {"type": "object"}},
            }
        ],
        "chat_id": "chat-1",
        "id": "msg-1",
    }

    payload = adapter.build_upstream_payload(
        form_data=form_data,
        endpoint_kind="responses",
        metadata={"chat_id": "chat-1"},
        strip_internal_keys=True,
        include_endpoint_kind=False,
    )

    assert "messages" not in payload
    assert payload["input"][0]["role"] == "user"
    assert payload["tools"][0]["name"] == "foo"
    assert "chat_id" not in payload
    assert payload["metadata"]["chat_id"] == "chat-1"


def test_build_upstream_payload_for_responses_prefers_canonical_form_data_over_base_payload():
    form_data = {
        "model": "custom-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1,
        "temperature": 0.2,
    }
    base_payload = {
        "model": "custom-model",
        "max_output_tokens": 1000,
        "temperature": 0.9,
        "future_unknown_field": {"x": 1},
    }

    payload = adapter.build_upstream_payload(
        form_data=form_data,
        endpoint_kind="responses",
        base_payload=base_payload,
        include_endpoint_kind=False,
    )

    assert payload["max_output_tokens"] == 1
    assert payload["temperature"] == 0.2
    assert payload["future_unknown_field"] == {"x": 1}


def test_build_upstream_payload_for_responses_uses_form_data_messages_not_base_input():
    form_data = {
        "model": "custom-model",
        "messages": [
            {"role": "system", "content": "model system"},
            {"role": "user", "content": "hello"},
        ],
    }
    base_payload = {
        "input": [{"role": "user", "content": "stale input"}],
    }

    payload = adapter.build_upstream_payload(
        form_data=form_data,
        endpoint_kind="responses",
        base_payload=base_payload,
        include_endpoint_kind=False,
    )

    assert payload["input"][0]["role"] == "system"
    assert payload["input"][0]["content"] == "model system"
    assert payload["input"][1]["role"] == "user"
    assert payload["input"][1]["content"] == "hello"


def test_build_upstream_payload_for_responses_normalizes_max_token_aliases():
    form_data = {
        "model": "custom-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 42,
    }
    base_payload = {
        "max_tokens": 99,
    }

    payload = adapter.build_upstream_payload(
        form_data=form_data,
        endpoint_kind="responses",
        base_payload=base_payload,
        include_endpoint_kind=False,
    )

    assert payload["max_output_tokens"] == 42
    assert "max_tokens" not in payload
    assert "max_completion_tokens" not in payload


def test_apply_prompt_cache_policy_keeps_explicit_retention_unchanged(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "resolve_prompt_cache_key_for_completion_request",
        lambda *_args, **_kwargs: "pc:v1:resolved",
    )

    payload = {"prompt_cache_key": "request-key", "prompt_cache_retention": "12h"}
    adapter.apply_prompt_cache_policy(
        provider_type="openai_responses",
        endpoint_kind="responses",
        payload=payload,
        metadata={"chat_id": "chat-1"},
        user=_user(),
    )

    assert payload["prompt_cache_key"] == "request-key"
    assert payload["prompt_cache_retention"] == "12h"


def test_resolve_prompt_cache_key_reads_and_persists_chat_metadata(monkeypatch):
    calls = []

    class DummyChats:
        def get_chat_by_id_and_user_id(self, *_args, **_kwargs):
            return SimpleNamespace(meta={})

        def get_chat_by_id(self, *_args, **_kwargs):
            return None

        def update_chat_metadata_by_id(self, chat_id, meta):
            calls.append((chat_id, meta))
            return SimpleNamespace(meta=meta)

    monkeypatch.setattr(adapter, "Chats", DummyChats())

    resolved = adapter.resolve_prompt_cache_key_for_completion_request(
        payload={}, metadata={"chat_id": "chat-1"}, user=_user()
    )
    assert resolved.startswith("pc:v1:")
    assert calls and calls[0][0] == "chat-1"
