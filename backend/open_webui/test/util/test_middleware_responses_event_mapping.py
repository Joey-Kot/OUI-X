from open_webui.utils import middleware


def _new_state():
    return {
        "text_emitted": False,
        "function_calls": {},
        "emitted_call_ids": set(),
        "call_indexes": {},
        "reasoning_summary_streamed": False,
        "reasoning_summary_current_text": "",
        "reasoning_summary_emitted_len": 0,
        "reasoning_summary_pending_separator": False,
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


def test_response_reasoning_summary_text_delta_maps_to_reasoning_content():
    payload = {
        "type": "response.reasoning_summary_text.delta",
        "delta": "streamed step",
    }

    mapped = middleware._map_responses_event_to_chat_chunk(payload, _new_state())

    assert mapped is not None
    assert mapped["choices"][0]["delta"]["reasoning_content"] == "streamed step"


def test_response_reasoning_summary_text_done_maps_when_no_delta_seen():
    payload = {
        "type": "response.reasoning_summary_text.done",
        "text": "done-only step",
    }

    mapped = middleware._map_responses_event_to_chat_chunk(payload, _new_state())

    assert mapped is not None
    assert mapped["choices"][0]["delta"]["reasoning_content"] == "done-only step"


def test_response_reasoning_summary_mult_part_inserts_newline_between_parts():
    state = _new_state()
    events = [
        {"type": "response.reasoning_summary_part.added"},
        {"type": "response.reasoning_summary_text.delta", "delta": "line 1"},
        {"type": "response.reasoning_summary_part.done"},
        {"type": "response.reasoning_summary_part.added"},
        {"type": "response.reasoning_summary_text.delta", "delta": "line 2"},
    ]

    chunks = []
    for payload in events:
        mapped = middleware._map_responses_event_to_chat_chunk(payload, state)
        if mapped and mapped.get("choices"):
            chunks.append(mapped["choices"][0]["delta"]["reasoning_content"])

    assert "".join(chunks) == "line 1\nline 2"


def test_response_completed_does_not_duplicate_reasoning_after_streaming_summary():
    state = _new_state()
    streamed = {
        "type": "response.reasoning_summary_text.delta",
        "delta": "streamed once",
    }
    completed = {
        "type": "response.completed",
        "response": {
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "streamed once"}],
                }
            ]
        },
    }

    mapped_streamed = middleware._map_responses_event_to_chat_chunk(streamed, state)
    mapped_completed = middleware._map_responses_event_to_chat_chunk(completed, state)

    assert mapped_streamed is not None
    assert mapped_streamed["choices"][0]["delta"]["reasoning_content"] == "streamed once"
    assert mapped_completed is None
