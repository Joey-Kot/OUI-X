from open_webui.routers.openai import (
    _responses_event_to_chat_chunks,
    chat_messages_to_responses_input,
    chat_to_responses_payload,
    extract_reasoning_content,
    responses_output_to_chat_tool_calls,
    responses_to_chat_compatible,
    sanitize_responses_metadata,
    to_responses_content_part,
)
import pytest
import json


def test_chat_messages_to_responses_input_supports_tool_followups():
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": '{"tz":"UTC"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"now":"x"}'},
    ]

    input_items = chat_messages_to_responses_input(messages)

    assert input_items[0] == {"role": "user", "content": "hello"}
    assert input_items[1]["type"] == "function_call"
    assert input_items[1]["call_id"] == "call_1"
    assert input_items[2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"now":"x"}',
    }


def test_chat_messages_to_responses_input_assistant_tool_call_content_is_output_text():
    messages = [
        {
            "role": "assistant",
            "content": "tool call preamble",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": '{"tz":"UTC"}'},
                }
            ],
        }
    ]

    input_items = chat_messages_to_responses_input(messages)
    assert input_items[1]["role"] == "assistant"
    assert input_items[1]["content"][0]["type"] == "output_text"


def test_chat_messages_to_responses_input_role_aware_text_conversion():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "u"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
    ]

    input_items = chat_messages_to_responses_input(messages)
    assert input_items[0]["content"][0]["type"] == "input_text"
    assert input_items[1]["content"][0]["type"] == "output_text"


def test_to_responses_content_part_role_normalization():
    assert to_responses_content_part("assistant", {"type": "input_text", "text": "x"}) == {
        "type": "output_text",
        "text": "x",
    }
    assert to_responses_content_part("user", {"type": "output_text", "text": "y"}) == {
        "type": "input_text",
        "text": "y",
    }
    assert to_responses_content_part("assistant", {"type": "refusal", "refusal": "n"}) == {
        "type": "refusal",
        "refusal": "n",
    }


def test_chat_to_responses_payload_maps_params_tools_and_reasoning():
    payload = {
        "model": "gpt-5",
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get time",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "get_time"}},
        "parallel_tool_calls": True,
        "max_tokens": 42,
        "reasoning_effort": "Medium",
        "verbosity": "LOW",
        "temperature": 0.3,
    }

    adapted = chat_to_responses_payload(
        payload,
        metadata={"chat_id": "abc"},
        api_config={"provider_type": "openai_responses"},
    )

    assert adapted["model"] == "gpt-5"
    assert adapted["stream"] is True
    assert adapted["input"] == [{"role": "user", "content": "hi"}]
    assert adapted["tools"][0]["name"] == "get_time"
    assert adapted["tool_choice"] == {"type": "function", "name": "get_time"}
    assert adapted["parallel_tool_calls"] is True
    assert adapted["max_output_tokens"] == 42
    assert adapted["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert adapted["verbosity"] == "low"
    assert "text" not in adapted or "verbosity" not in adapted.get("text", {})
    assert adapted["temperature"] == 0.3
    assert "metadata" not in adapted


def test_chat_to_responses_payload_drops_removed_legacy_sampling_params():
    payload = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "hi"}],
        "min_p": 0.1,
        "repeat_penalty": 1.2,
        "tfs_z": 1.0,
        "repeat_last_n": 64,
        "mirostat_tau": 5,
        "mirostat_eta": 0.1,
        "mirostat": 2,
        "use_mmap": True,
        "use_mlock": True,
    }

    adapted = chat_to_responses_payload(
        payload, metadata=None, api_config={"provider_type": "openai_responses"}
    )

    for key in [
        "min_p",
        "repeat_penalty",
        "tfs_z",
        "repeat_last_n",
        "mirostat_tau",
        "mirostat_eta",
        "mirostat",
        "use_mmap",
        "use_mlock",
    ]:
        assert key not in adapted


def test_chat_to_responses_payload_keeps_reasoning_none_and_omits_empty_verbosity():
    payload = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "none",
        "verbosity": "",
    }

    adapted = chat_to_responses_payload(
        payload, metadata=None, api_config={"provider_type": "openai_responses"}
    )

    assert adapted["reasoning"] == {"effort": "none", "summary": "auto"}
    assert "verbosity" not in adapted

def test_chat_to_responses_payload_custom_summary_overrides_default():
    payload = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "hi"}],
        "summary": "detailed",
    }

    adapted = chat_to_responses_payload(
        payload, metadata=None, api_config={"provider_type": "openai_responses"}
    )

    assert adapted["reasoning"]["summary"] == "detailed"


def test_chat_to_responses_payload_reasoning_summary_has_priority_over_top_level_summary():
    payload = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "hi"}],
        "summary": "detailed",
        "reasoning": {"summary": "concise"},
    }

    adapted = chat_to_responses_payload(
        payload, metadata=None, api_config={"provider_type": "openai_responses"}
    )

    assert adapted["reasoning"]["summary"] == "concise"


def test_chat_to_responses_payload_invalid_summary_falls_back_to_auto():
    payload = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "hi"}],
        "summary": "foo",
    }

    adapted = chat_to_responses_payload(
        payload, metadata=None, api_config={"provider_type": "openai_responses"}
    )

    assert adapted["reasoning"]["summary"] == "auto"


def test_sanitize_responses_metadata_removes_non_serializable_values():
    metadata = {
        "chat_id": "chat_1",
        "nested": {
            "keep": 1,
            "drop": object(),
        },
        "items": ["ok", object(), 2],
    }

    sanitized = sanitize_responses_metadata(metadata)

    assert sanitized == {
        "chat_id": "chat_1",
        "nested": {"keep": 1},
        "items": ["ok", 2],
    }
    json.dumps(sanitized)


def test_chat_to_responses_payload_does_not_forward_internal_runtime_metadata():
    payload = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "hello"}],
    }

    adapted = chat_to_responses_payload(
        payload,
        metadata={"mcp_clients": object(), "chat_id": "chat_1"},
        api_config={"provider_type": "openai_responses"},
    )

    assert "metadata" not in adapted
    json.dumps(adapted)


def test_responses_output_to_chat_tool_calls_preserves_call_id():
    output = [
        {
            "type": "function_call",
            "call_id": "call_abc",
            "name": "tool_x",
            "arguments": '{"a":1}',
        }
    ]

    tool_calls = responses_output_to_chat_tool_calls(output)

    assert tool_calls == [
        {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "tool_x", "arguments": '{"a":1}'},
        }
    ]


def test_responses_to_chat_compatible_maps_message_tools_and_reasoning_summary():
    resp = {
        "id": "resp_1",
        "created_at": 1,
        "model": "gpt-5",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_time",
                "arguments": '{"tz":"UTC"}',
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "final answer"}],
            },
        ],
        "reasoning": {"summary": "thinking summary"},
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 1},
            "total_tokens": 15,
            "custom_metric": 99,
        },
    }

    out = responses_to_chat_compatible(resp)

    assert out["choices"][0]["message"]["content"] == "final answer"
    assert out["choices"][0]["message"]["reasoning_content"] == "thinking summary"
    assert out["choices"][0]["message"]["tool_calls"][0]["id"] == "call_1"
    assert out["usage"] == {
        "input_tokens": 10,
        "input_tokens_details": {"cached_tokens": 3},
        "output_tokens": 5,
        "output_tokens_details": {"reasoning_tokens": 1},
        "custom_metric": 99,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


def test_extract_reasoning_content_prefers_output_reasoning_over_root_summary():
    resp = {
        "reasoning": {"summary": "detailed"},
        "output": [
            {
                "type": "reasoning",
                "summary": [
                    {"text": "actual summary 1"},
                    {"text": "actual summary 2"},
                ],
            }
        ],
    }

    assert extract_reasoning_content(resp) == "actual summary 1\nactual summary 2"


def test_extract_reasoning_content_falls_back_to_root_reasoning_summary():
    resp = {
        "reasoning": {"summary": [{"text": "fallback summary"}]},
        "output": [],
    }

    assert extract_reasoning_content(resp) == "fallback summary"


def test_responses_event_mapping_emits_reasoning_tool_and_done():
    state = {
        "function_calls": {},
        "call_indexes": {},
        "emitted_call_ids": set(),
        "text_emitted": False,
    }

    output_done_chunks = _responses_event_to_chat_chunks(
        "response.output_item.done",
        {
            "item": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_time",
                "arguments": '{"tz":"UTC"}',
            }
        },
        state,
    )

    completed_chunks = _responses_event_to_chat_chunks(
        "response.completed",
        {
            "response": {
                "reasoning": {"summary": "detailed"},
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{"text": "thinking..."}],
                    },
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "hello"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
        },
        state,
    )

    text = "".join(output_done_chunks + completed_chunks)
    assert "reasoning_content" in text
    assert "thinking..." in text
    assert "detailed" not in text
    assert "tool_calls" in text
    assert '"content": "hello"' in text
    assert '"input_tokens": 1' in text
    assert '"prompt_tokens": 1' in text
    assert "[DONE]" in text


def test_responses_event_mapping_usage_falls_back_total_tokens_when_missing():
    state = {
        "function_calls": {},
        "call_indexes": {},
        "emitted_call_ids": set(),
        "text_emitted": False,
    }

    completed_chunks = _responses_event_to_chat_chunks(
        "response.completed",
        {
            "response": {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "hello"}],
                    }
                ],
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 3,
                    "input_tokens_details": {"cached_tokens": 1},
                },
            }
        },
        state,
    )

    text = "".join(completed_chunks)
    assert '"input_tokens_details": {"cached_tokens": 1}' in text
    assert '"prompt_tokens": 2' in text
    assert '"completion_tokens": 3' in text
    assert '"total_tokens": 5' in text
