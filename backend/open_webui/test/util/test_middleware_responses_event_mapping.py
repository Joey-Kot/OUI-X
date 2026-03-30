from open_webui.utils import middleware


def _new_state():
    return {
        "text_emitted": False,
        "function_calls": {},
        "emitted_call_ids": set(),
        "call_indexes": {},
    }


def test_response_completed_ignores_root_reasoning_summary_config_string():
    payload = {
        "type": "response.completed",
        "response": {
            "output": [],
            "reasoning": {"summary": "detailed"},
        },
    }

    mapped = middleware._map_responses_event_to_chat_chunk(payload, _new_state())

    assert mapped is None


def test_response_completed_maps_reasoning_from_output_summary_text():
    payload = {
        "type": "response.completed",
        "response": {
            "output": [
                {
                    "type": "reasoning",
                    "summary": [
                        {"type": "summary_text", "text": "step 1"},
                        {"type": "summary_text", "text": "step 2"},
                    ],
                }
            ]
        },
    }

    mapped = middleware._map_responses_event_to_chat_chunk(payload, _new_state())

    assert mapped is not None
    assert mapped["choices"][0]["delta"]["reasoning_content"] == "step 1\nstep 2"


def test_response_completed_keeps_text_fallback_and_unresolved_tool_calls():
    payload = {
        "type": "response.completed",
        "response": {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "fallback text"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup_weather",
                    "arguments": '{"city":"Shanghai"}',
                },
            ]
        },
    }

    mapped = middleware._map_responses_event_to_chat_chunk(payload, _new_state())

    assert mapped is not None
    delta = mapped["choices"][0]["delta"]
    assert delta["content"] == "fallback text"
    assert delta["tool_calls"] == [
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "arguments": '{"city":"Shanghai"}',
            },
        }
    ]
