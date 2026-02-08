import asyncio

import pytest

from open_webui.utils.tool_orchestrator import (
    build_followup_messages,
    execute_tool_calls_parallel,
    parse_tool_arguments,
)
from open_webui.utils.tools_runtime import normalize_function_schema


def test_normalize_function_schema_invalid_values():
    normalized = normalize_function_schema(
        {
            "type": "array",
            "properties": {"q": {"type": "str"}, "x": "invalid"},
            "required": "q",
            "additionalProperties": "invalid",
        }
    )

    assert normalized["type"] == "object"
    assert normalized["properties"]["q"]["type"] == "string"
    assert normalized["properties"]["x"] == {"type": "string"}
    assert normalized["required"] == []
    assert normalized["additionalProperties"] is False


def test_parse_tool_arguments_fallbacks():
    assert parse_tool_arguments('{"a": 1}') == {"a": 1}
    assert parse_tool_arguments("{'b': 2}") == {"b": 2}
    assert parse_tool_arguments("invalid") == {}
    assert parse_tool_arguments("") == {}


@pytest.mark.asyncio
async def test_execute_tool_calls_parallel_preserves_input_order_and_concurrency_limit():
    active = 0
    max_seen = 0
    lock = asyncio.Lock()

    async def slow_tool(delay: float, value: str):
        nonlocal active, max_seen
        async with lock:
            active += 1
            max_seen = max(max_seen, active)
        await asyncio.sleep(delay)
        async with lock:
            active -= 1
        return {"value": value}

    async def run_one(**kwargs):
        return await slow_tool(**kwargs)

    async def run_two(**kwargs):
        return await slow_tool(**kwargs)

    tools = {
        "one": {
            "callable": run_one,
            "spec": {
                "parameters": {
                    "type": "object",
                    "properties": {"delay": {}, "value": {}},
                }
            },
            "type": "local",
            "direct": False,
        },
        "two": {
            "callable": run_two,
            "spec": {
                "parameters": {
                    "type": "object",
                    "properties": {"delay": {}, "value": {}},
                }
            },
            "type": "local",
            "direct": False,
        },
    }

    tool_calls = [
        {
            "id": "call-1",
            "function": {
                "name": "one",
                "arguments": '{"delay": 0.02, "value": "first"}',
            },
        },
        {
            "id": "call-2",
            "function": {
                "name": "two",
                "arguments": '{"delay": 0.01, "value": "second"}',
            },
        },
    ]

    def process_tool_result(*args, **_kwargs):
        raw_result = args[2]
        return str(raw_result), [], []

    outcomes = await execute_tool_calls_parallel(
        tool_calls=tool_calls,
        tools=tools,
        max_concurrency=1,
        event_caller=None,
        form_data={"messages": []},
        metadata={"files": []},
        request=None,
        user=None,
        process_tool_result=process_tool_result,
    )

    assert [outcome.tool_call_id for outcome in outcomes] == ["call-1", "call-2"]
    assert max_seen == 1

    followups = build_followup_messages(
        [], {"role": "assistant", "tool_calls": tool_calls}, outcomes
    )
    assert followups[0]["role"] == "assistant"
    assert followups[1]["role"] == "tool"
    assert followups[1]["tool_call_id"] == "call-1"
