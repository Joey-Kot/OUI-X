import asyncio

import pytest

from open_webui.utils.tool_orchestrator import (
    build_followup_messages,
    clamp_max_tool_calls_per_round,
    clamp_tool_timeout_seconds,
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


def test_tool_calling_clamp_defaults_and_bounds():
    assert clamp_tool_timeout_seconds("") == 60
    assert clamp_tool_timeout_seconds("oops") == 60
    assert clamp_tool_timeout_seconds(-1) == 1
    assert clamp_tool_timeout_seconds(1000) == 600

    assert clamp_max_tool_calls_per_round("") == 20
    assert clamp_max_tool_calls_per_round("oops") == 20
    assert clamp_max_tool_calls_per_round(0) == 1
    assert clamp_max_tool_calls_per_round(1000) == 100


@pytest.mark.asyncio
async def test_execute_tool_calls_parallel_timeout_and_max_limit():
    async def slow_tool(**_kwargs):
        await asyncio.sleep(0.1)
        return {"ok": True}

    tools = {
        "one": {
            "callable": slow_tool,
            "spec": {"parameters": {"type": "object", "properties": {}}},
            "type": "local",
            "source": "local",
            "scope": None,
            "direct": False,
        },
        "two": {
            "callable": slow_tool,
            "spec": {"parameters": {"type": "object", "properties": {}}},
            "type": "local",
            "source": "local",
            "scope": None,
            "direct": False,
        },
    }
    tool_calls = [
        {"id": "call-1", "function": {"name": "one", "arguments": "{}"}},
        {"id": "call-2", "function": {"name": "two", "arguments": "{}"}},
    ]

    def process_tool_result(*args, **_kwargs):
        return str(args[2]), [], []

    outcomes = await execute_tool_calls_parallel(
        tool_calls=tool_calls,
        tools=tools,
        max_concurrency=2,
        event_caller=None,
        form_data={"messages": []},
        metadata={
            "tool_calling_config": {
                "global": {
                    "tool_call_timeout_seconds": 1,
                    "max_tool_calls_per_round": 1,
                }
            }
        },
        request=None,
        user=None,
        process_tool_result=process_tool_result,
        tool_timeout_seconds=0,
        max_tool_calls_per_round=1,
        user_override_policy="whole_round",
    )

    assert outcomes[0].tool_call_id == "call-1"
    assert "tool execution timed out" in outcomes[0].content
    assert outcomes[1].tool_call_id == "call-2"
    assert "tool_skipped_max_calls" in outcomes[1].content


@pytest.mark.asyncio
async def test_user_mcp_override_max_calls_applies_to_whole_round():
    async def fast_tool(**_kwargs):
        return {"ok": True}

    tools = {
        "local_tool": {
            "callable": fast_tool,
            "spec": {"parameters": {"type": "object", "properties": {}}},
            "type": "local",
            "source": "local",
            "scope": None,
            "direct": False,
        },
        "user_mcp_tool": {
            "callable": fast_tool,
            "spec": {"parameters": {"type": "object", "properties": {}}},
            "type": "mcp",
            "source": "mcp",
            "scope": "user",
            "direct": False,
        },
    }

    tool_calls = [
        {"id": "call-1", "function": {"name": "local_tool", "arguments": "{}"}},
        {"id": "call-2", "function": {"name": "user_mcp_tool", "arguments": "{}"}},
    ]

    def process_tool_result(*args, **_kwargs):
        return str(args[2]), [], []

    outcomes = await execute_tool_calls_parallel(
        tool_calls=tool_calls,
        tools=tools,
        max_concurrency=2,
        event_caller=None,
        form_data={"messages": []},
        metadata={
            "tool_calling_config": {
                "global": {"tool_call_timeout_seconds": 60, "max_tool_calls_per_round": 20},
                "user": {"tool_call_timeout_seconds": 30, "max_tool_calls_per_round": 1},
            }
        },
        request=None,
        user=None,
        process_tool_result=process_tool_result,
        user_override_policy="whole_round",
    )

    assert outcomes[0].tool_call_id == "call-1"
    assert "tool_skipped_max_calls" in outcomes[1].content
