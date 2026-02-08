import ast
import asyncio
import json
import logging
import time
from typing import Any, Callable

from open_webui.env import CHAT_RESPONSE_MAX_TOOL_CALL_RETRIES
from open_webui.utils.tools import get_updated_tool_function
from open_webui.utils.tools_runtime import ToolExecutionOutcome

log = logging.getLogger(__name__)


def parse_tool_arguments(arguments: Any) -> dict:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}

    try:
        parsed = ast.literal_eval(arguments)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        parsed = json.loads(arguments)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {}


def sanitize_tool_calls(tool_calls: list[dict]) -> list[dict]:
    for tool_call in tool_calls:
        args = parse_tool_arguments(tool_call.get("function", {}).get("arguments", "{}"))
        tool_call.setdefault("function", {})["arguments"] = json.dumps(args)
    return tool_calls


def extract_tool_calls(assistant_message: dict | None) -> list[dict]:
    if not assistant_message:
        return []
    tool_calls = assistant_message.get("tool_calls") or []
    return tool_calls if isinstance(tool_calls, list) else []


def extract_assistant_message_from_non_stream(response: dict) -> dict:
    choices = response.get("choices") or []
    if not choices:
        return {"role": "assistant", "content": ""}
    return choices[0].get("message", {}) or {"role": "assistant", "content": ""}


async def extract_assistant_message_from_stream(response) -> dict:
    assistant_message = {"role": "assistant", "content": "", "tool_calls": []}

    if not hasattr(response, "body_iterator"):
        return assistant_message

    tool_calls_by_index = {}
    async for chunk in response.body_iterator:
        data = json.loads(chunk.decode("utf-8", "replace"))
        delta = ((data.get("choices") or [{}])[0] or {}).get("delta", {})
        if delta.get("content"):
            assistant_message["content"] += delta["content"]

        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index")
            if idx is None:
                continue
            current = tool_calls_by_index.setdefault(
                idx,
                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if tc.get("id"):
                current["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                current["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                current["function"]["arguments"] += fn["arguments"]

    if tool_calls_by_index:
        assistant_message["tool_calls"] = [
            tool_calls_by_index[idx] for idx in sorted(tool_calls_by_index.keys())
        ]

    if response.background is not None:
        await response.background()

    return assistant_message


async def _execute_single_tool_call(
    *,
    tool_call: dict,
    tools: dict[str, dict],
    max_concurrency_sem: asyncio.Semaphore,
    event_caller,
    form_data: dict,
    metadata: dict,
    request,
    user,
    process_tool_result: Callable,
) -> ToolExecutionOutcome:
    async with max_concurrency_sem:
        tool_call_id = tool_call.get("id", "")
        tool_function_name = tool_call.get("function", {}).get("name", "")
        tool_function_params = parse_tool_arguments(
            tool_call.get("function", {}).get("arguments", "{}")
        )

        start = time.perf_counter()
        tool_result = None
        tool = tools.get(tool_function_name)
        tool_type = None
        direct_tool = False
        error = None

        if tool is None:
            error = f"Tool {tool_function_name} not found"
        else:
            spec = tool.get("spec", {})
            tool_type = tool.get("type", "")
            direct_tool = tool.get("direct", False)

            allowed_params = spec.get("parameters", {}).get("properties", {}).keys()
            tool_function_params = {
                key: value
                for key, value in tool_function_params.items()
                if key in allowed_params
            }

            try:
                if direct_tool:
                    tool_result = await event_caller(
                        {
                            "type": "execute:tool",
                            "data": {
                                "id": tool_call_id,
                                "name": tool_function_name,
                                "params": tool_function_params,
                                "server": tool.get("server", {}),
                                "session_id": metadata.get("session_id", None),
                            },
                        }
                    )
                else:
                    tool_function = get_updated_tool_function(
                        function=tool["callable"],
                        extra_params={
                            "__messages__": form_data.get("messages", []),
                            "__files__": metadata.get("files", []),
                        },
                    )
                    tool_result = await tool_function(**tool_function_params)
            except Exception as exc:
                error = str(exc)
                tool_result = {
                    "error": error,
                    "tool": tool_function_name,
                    "ok": False,
                }

        content, files, embeds = process_tool_result(
            request,
            tool_function_name,
            tool_result,
            tool_type,
            direct_tool,
            metadata,
            user,
        )

        duration_ms = int((time.perf_counter() - start) * 1000)
        log.info(
            "tool_call_id=%s tool_name=%s source=%s duration_ms=%s ok=%s",
            tool_call_id,
            tool_function_name,
            tool_type or "unknown",
            duration_ms,
            error is None,
        )

        return ToolExecutionOutcome(
            tool_call_id=tool_call_id,
            name=tool_function_name,
            ok=error is None,
            content=content or "",
            files=files or [],
            embeds=embeds or [],
            error=error,
        )


async def execute_tool_calls_parallel(
    *,
    tool_calls: list[dict],
    tools: dict[str, dict],
    max_concurrency: int,
    event_caller,
    form_data: dict,
    metadata: dict,
    request,
    user,
    process_tool_result: Callable,
) -> list[ToolExecutionOutcome]:
    if not tool_calls:
        return []

    sanitized_calls = sanitize_tool_calls(tool_calls)
    sem = asyncio.Semaphore(max(1, max_concurrency))

    tasks = [
        _execute_single_tool_call(
            tool_call=tool_call,
            tools=tools,
            max_concurrency_sem=sem,
            event_caller=event_caller,
            form_data=form_data,
            metadata=metadata,
            request=request,
            user=user,
            process_tool_result=process_tool_result,
        )
        for tool_call in sanitized_calls
    ]
    outcomes = await asyncio.gather(*tasks)

    outcomes_by_id = {outcome.tool_call_id: outcome for outcome in outcomes}
    ordered = [
        outcomes_by_id.get(tool_call.get("id", ""))
        for tool_call in sanitized_calls
        if outcomes_by_id.get(tool_call.get("id", ""))
    ]
    return ordered


def build_followup_messages(
    prior_messages: list[dict], assistant_message: dict, outcomes: list[ToolExecutionOutcome]
) -> list[dict]:
    tool_messages = [
        {
            "role": "tool",
            "tool_call_id": outcome.tool_call_id,
            "content": outcome.content,
        }
        for outcome in outcomes
    ]
    return [*prior_messages, assistant_message, *tool_messages]


def should_continue_loop(tool_calls: list[dict], retries: int) -> bool:
    return bool(tool_calls) and retries < CHAT_RESPONSE_MAX_TOOL_CALL_RETRIES


async def run_native_tool_loop(
    *,
    initial_form_data: dict,
    call_model: Callable[[dict], Any],
    extract_message: Callable[[Any], Any],
    execute_calls: Callable[[list[dict], dict], Any],
    max_retries: int = CHAT_RESPONSE_MAX_TOOL_CALL_RETRIES,
):
    form_data = dict(initial_form_data)
    retries = 0

    while retries < max_retries:
        response = await call_model(form_data)
        assistant_message = await extract_message(response)
        tool_calls = extract_tool_calls(assistant_message)

        if not tool_calls:
            return response, assistant_message

        outcomes = await execute_calls(tool_calls, form_data)
        form_data["messages"] = build_followup_messages(
            prior_messages=form_data.get("messages", []),
            assistant_message=assistant_message,
            outcomes=outcomes,
        )
        retries += 1

    return None, {"role": "assistant", "content": "", "tool_calls": []}
