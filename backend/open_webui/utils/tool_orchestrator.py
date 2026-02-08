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

DEFAULT_TOOL_CALL_TIMEOUT_SECONDS = 60
DEFAULT_MAX_TOOL_CALLS_PER_ROUND = 20


def clamp_tool_timeout_seconds(value: Any) -> int:
    try:
        normalized = int(value)
    except Exception:
        normalized = DEFAULT_TOOL_CALL_TIMEOUT_SECONDS
    return max(1, min(600, normalized))


def clamp_max_tool_calls_per_round(value: Any) -> int:
    try:
        normalized = int(value)
    except Exception:
        normalized = DEFAULT_MAX_TOOL_CALLS_PER_ROUND
    return max(1, min(100, normalized))


def _is_user_scoped_mcp_tool(tool: dict | None) -> bool:
    if not tool:
        return False
    return tool.get("source") == "mcp" and tool.get("scope") == "user"


def _get_tool_calling_configs(metadata: dict | None) -> tuple[dict, dict]:
    config = (metadata or {}).get("tool_calling_config", {}) if isinstance(metadata, dict) else {}
    global_config = config.get("global", {}) if isinstance(config, dict) else {}
    user_config = config.get("user", {}) if isinstance(config, dict) else {}
    return (
        global_config if isinstance(global_config, dict) else {},
        user_config if isinstance(user_config, dict) else {},
    )


def _resolve_timeout_for_tool(
    tool: dict | None,
    metadata: dict,
    fallback_timeout_seconds: int,
) -> int:
    global_config, user_config = _get_tool_calling_configs(metadata)
    timeout_seconds = clamp_tool_timeout_seconds(
        global_config.get("tool_call_timeout_seconds", fallback_timeout_seconds)
    )

    if _is_user_scoped_mcp_tool(tool) and "tool_call_timeout_seconds" in user_config:
        timeout_seconds = clamp_tool_timeout_seconds(
            user_config.get("tool_call_timeout_seconds")
        )

    return timeout_seconds


def _resolve_effective_max_calls_for_round(
    *,
    tool_calls: list[dict],
    tools: dict[str, dict],
    metadata: dict,
    fallback_max_calls: int,
    user_override_policy: str,
) -> int:
    global_config, user_config = _get_tool_calling_configs(metadata)
    effective_max = clamp_max_tool_calls_per_round(
        global_config.get("max_tool_calls_per_round", fallback_max_calls)
    )

    user_max = user_config.get("max_tool_calls_per_round")
    if user_max is None:
        return effective_max

    if user_override_policy != "whole_round":
        return effective_max

    has_user_mcp_call = False
    for tool_call in tool_calls:
        tool_name = tool_call.get("function", {}).get("name", "")
        if _is_user_scoped_mcp_tool(tools.get(tool_name)):
            has_user_mcp_call = True
            break

    if has_user_mcp_call:
        return clamp_max_tool_calls_per_round(user_max)

    return effective_max


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
    timeout_seconds: int,
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
                    execution_coro = event_caller(
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

                    execution_coro = tool_function(**tool_function_params)

                tool_result = await asyncio.wait_for(execution_coro, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                error = f"tool execution timed out after {timeout_seconds}s"
                tool_result = {
                    "ok": False,
                    "error": error,
                    "tool": tool_function_name,
                    "code": "tool_timeout",
                }
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
    tool_timeout_seconds: int = DEFAULT_TOOL_CALL_TIMEOUT_SECONDS,
    max_tool_calls_per_round: int = DEFAULT_MAX_TOOL_CALLS_PER_ROUND,
    user_override_policy: str = "whole_round",
) -> list[ToolExecutionOutcome]:
    if not tool_calls:
        return []

    sanitized_calls = sanitize_tool_calls(tool_calls)
    effective_max_calls = _resolve_effective_max_calls_for_round(
        tool_calls=sanitized_calls,
        tools=tools,
        metadata=metadata,
        fallback_max_calls=max_tool_calls_per_round,
        user_override_policy=user_override_policy,
    )

    executable_calls = sanitized_calls[:effective_max_calls]
    skipped_calls = sanitized_calls[effective_max_calls:]

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
            timeout_seconds=_resolve_timeout_for_tool(
                tools.get(tool_call.get("function", {}).get("name", "")),
                metadata,
                tool_timeout_seconds,
            ),
        )
        for tool_call in executable_calls
    ]
    outcomes = await asyncio.gather(*tasks)

    for skipped_call in skipped_calls:
        skipped_tool_name = skipped_call.get("function", {}).get("name", "")
        skipped_id = skipped_call.get("id", "")
        skipped_error = (
            f"Tool calls were skipped because the maximum number of calls per round for a single tool ({effective_max_calls}) had been reached."
        )
        outcomes.append(
            ToolExecutionOutcome(
                tool_call_id=skipped_id,
                name=skipped_tool_name,
                ok=False,
                content=json.dumps(
                    {
                        "ok": False,
                        "error": skipped_error,
                        "tool": skipped_tool_name,
                        "code": "tool_skipped_max_calls",
                    },
                    ensure_ascii=False,
                ),
                error=skipped_error,
            )
        )

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
