import asyncio
import hashlib
import json
import logging
from typing import Any, Optional

import aiohttp
from aiocache import cached
import requests

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from fastapi import Depends, HTTPException, Request, APIRouter
from fastapi.responses import (
    FileResponse,
    StreamingResponse,
    JSONResponse,
    PlainTextResponse,
)
from pydantic import BaseModel
from starlette.background import BackgroundTask

from open_webui.models.models import Models
from open_webui.config import (
    CACHE_DIR,
)
from open_webui.env import (
    MODELS_CACHE_TTL,
    AIOHTTP_CLIENT_SESSION_SSL,
    AIOHTTP_CLIENT_TIMEOUT,
    AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST,
    ENABLE_FORWARD_USER_INFO_HEADERS,
    BYPASS_MODEL_ACCESS_CONTROL,
)
from open_webui.models.users import UserModel

from open_webui.constants import ERROR_MESSAGES


from open_webui.utils.payload import (
    apply_model_params_to_body_openai,
    apply_system_prompt_to_body,
)
from open_webui.utils.misc import (
    convert_logit_bias_input_to_json,
    stream_chunks_handler,
)

from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import has_access
from open_webui.utils.headers import include_user_info_headers


log = logging.getLogger(__name__)


##########################################
#
# Utility functions
#
##########################################


async def send_get_request(url, key=None, user: UserModel = None):
    timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST)
    try:
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            headers = {
                **({"Authorization": f"Bearer {key}"} if key else {}),
            }

            if ENABLE_FORWARD_USER_INFO_HEADERS and user:
                headers = include_user_info_headers(headers, user)

            async with session.get(
                url,
                headers=headers,
                ssl=AIOHTTP_CLIENT_SESSION_SSL,
            ) as response:
                return await response.json()
    except Exception as e:
        # Handle connection error here
        log.error(f"Connection error: {e}")
        return None


async def cleanup_response(
    response: Optional[aiohttp.ClientResponse],
    session: Optional[aiohttp.ClientSession],
):
    if response:
        response.close()
    if session:
        await session.close()


def openai_reasoning_model_handler(payload):
    """
    Handle reasoning model specific parameters
    """
    if "max_tokens" in payload:
        # Convert "max_tokens" to "max_completion_tokens" for all reasoning models
        payload["max_completion_tokens"] = payload["max_tokens"]
        del payload["max_tokens"]

    # Handle system role conversion based on model type
    if payload["messages"][0]["role"] == "system":
        model_lower = payload["model"].lower()
        # Legacy models use "user" role instead of "system"
        if model_lower.startswith("o1-mini") or model_lower.startswith("o1-preview"):
            payload["messages"][0]["role"] = "user"
        else:
            payload["messages"][0]["role"] = "developer"

    return payload


def get_provider_type(api_config: Optional[dict]) -> str:
    if not isinstance(api_config, dict):
        return "openai"

    provider_type = api_config.get("provider_type")
    if provider_type in {"openai", "azure_openai", "openai_responses"}:
        return provider_type

    if api_config.get("azure", False):
        return "azure_openai"

    return "openai"


def is_responses_provider(api_config: Optional[dict]) -> bool:
    return get_provider_type(api_config) == "openai_responses"


def _normalize_tool_choice_for_responses(tool_choice):
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function" and isinstance(
            tool_choice.get("function"), dict
        ):
            return {
                "type": "function",
                "name": tool_choice.get("function", {}).get("name"),
            }
        return tool_choice

    if isinstance(tool_choice, str):
        return tool_choice

    return None


ALLOWED_REASONING_SUMMARY_VALUES = {"auto", "concise", "detailed"}


def normalize_reasoning_summary(
    value: Any,
    default: str = "auto",
    source: str = "reasoning.summary",
) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ALLOWED_REASONING_SUMMARY_VALUES:
            return normalized
        if normalized:
            log.debug(
                "Invalid Responses %s value %s. Falling back to %s.",
                source,
                value,
                default,
            )
        return default

    if value is not None:
        log.debug(
            "Invalid Responses %s type %s. Falling back to %s.",
            source,
            type(value).__name__,
            default,
        )

    return default


def _normalize_tools_for_responses(tools: list) -> list:
    normalized_tools = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue

        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            function = tool.get("function")
            normalized_tools.append(
                {
                    "type": "function",
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "parameters": function.get(
                        "parameters",
                        {
                            "type": "object",
                            "properties": {},
                        },
                    ),
                }
            )
            continue

        normalized_tools.append(tool)

    return normalized_tools


def to_responses_content_part(role: str, part: dict) -> dict | None:
    if not isinstance(part, dict):
        return None

    part_type = part.get("type")
    user_like_roles = {"user", "system", "developer"}

    if part_type == "text":
        target_type = "output_text" if role == "assistant" else "input_text"
        text_value = part.get("text") or part.get("content", "")
        return {"type": target_type, "text": text_value}

    if part_type == "image_url":
        image_url = part.get("image_url", {})
        return {
            "type": "input_image",
            "image_url": image_url.get("url", "")
            if isinstance(image_url, dict)
            else image_url,
        }

    if part_type == "refusal":
        # Assistant-only content block in responses API.
        return part if role == "assistant" else None

    if part_type in {"input_image", "input_file"}:
        return part if role in user_like_roles else None

    if part_type == "input_text":
        if role == "assistant":
            return {"type": "output_text", "text": part.get("text", "")}
        return part

    if part_type == "output_text":
        if role == "assistant":
            return part
        return {"type": "input_text", "text": part.get("text", "")}

    return None


def chat_messages_to_responses_input(messages: list[dict]) -> list:
    input_items = []

    for message in messages or []:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "user")

        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": message.get("content", ""),
                }
            )
            continue

        tool_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            for tool_call in tool_calls:
                function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                arguments = function.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)

                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.get("id", ""),
                        "name": function.get("name", ""),
                        "arguments": arguments,
                    }
                )

            if message.get("content"):
                input_items.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": message.get("content", ""),
                            }
                        ],
                    }
                )
            continue

        content = message.get("content", "")
        if isinstance(content, str):
            input_items.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            converted_content = []
            for part in content:
                if not isinstance(part, dict):
                    continue

                normalized_part = to_responses_content_part(role=role, part=part)
                if normalized_part:
                    converted_content.append(normalized_part)

            if converted_content:
                input_items.append({"role": role, "content": converted_content})

    return input_items


def responses_input_from_tool_followups(messages: list[dict]) -> list:
    return chat_messages_to_responses_input(messages)


def _extract_reasoning_summary(reasoning: dict | None) -> str:
    if not isinstance(reasoning, dict):
        return ""

    summary = reasoning.get("summary")
    if isinstance(summary, str):
        normalized = summary.strip().lower()
        # Some providers return the summary mode here (auto/concise/detailed)
        # rather than the actual reasoning text; don't surface these in UI.
        if normalized in ALLOWED_REASONING_SUMMARY_VALUES:
            return ""
        return summary

    if isinstance(summary, list):
        texts = []
        for item in summary:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text = item.get("text", "")
                if text.strip().lower() in ALLOWED_REASONING_SUMMARY_VALUES:
                    continue
                texts.append(text)
        return "\n".join(texts)

    return ""


def extract_reasoning_text_from_output(output_items: list[dict] | None) -> str:
    if not isinstance(output_items, list):
        return ""

    texts: list[str] = []
    for item in output_items:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue

        summary = item.get("summary")
        if isinstance(summary, list):
            for summary_item in summary:
                if isinstance(summary_item, dict) and isinstance(
                    summary_item.get("text"), str
                ):
                    text = summary_item.get("text", "").strip()
                    if not text:
                        continue
                    if text.lower() in ALLOWED_REASONING_SUMMARY_VALUES:
                        continue
                    texts.append(text)

    return "\n".join(texts)


def extract_reasoning_content(resp: dict | None) -> str:
    if not isinstance(resp, dict):
        return ""

    reasoning_from_output = extract_reasoning_text_from_output(resp.get("output"))
    if reasoning_from_output:
        return reasoning_from_output

    return _extract_reasoning_summary(resp.get("reasoning"))


def _normalize_responses_usage(usage: dict | None) -> dict | None:
    if not isinstance(usage, dict):
        return None

    return {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def _responses_event_to_chat_chunks(
    event_type: str,
    payload: dict | None,
    state: dict,
) -> list[str]:
    chunks: list[str] = []
    payload = payload if isinstance(payload, dict) else {}

    def emit(data: dict):
        chunks.append(f"data: {json.dumps(data)}\n\n")

    if event_type == "response.output_text.delta":
        delta = payload.get("delta", "")
        if delta:
            state["text_emitted"] = True
            emit({"choices": [{"delta": {"content": delta}}]})

    elif event_type in {
        "response.function_call_arguments.delta",
        "response.function_call.delta",
    }:
        call_id = payload.get("call_id") or payload.get("item_id")
        if call_id:
            function_state = state["function_calls"].setdefault(
                call_id, {"name": payload.get("name", ""), "arguments": ""}
            )

            if payload.get("name"):
                function_state["name"] = payload.get("name")

            arguments_delta = payload.get("delta", "")
            if isinstance(arguments_delta, str) and arguments_delta:
                function_state["arguments"] += arguments_delta

    elif event_type == "response.output_item.done":
        item = payload.get("item", {})
        if isinstance(item, dict) and item.get("type") == "function_call":
            call_id = item.get("call_id") or item.get("id", "")
            if call_id and call_id not in state["emitted_call_ids"]:
                function_state = state["function_calls"].setdefault(
                    call_id,
                    {
                        "name": item.get("name", ""),
                        "arguments": "",
                    },
                )

                if item.get("name"):
                    function_state["name"] = item.get("name")

                arguments = item.get("arguments", "")
                if isinstance(arguments, str):
                    function_state["arguments"] = arguments

                index = state["call_indexes"].setdefault(
                    call_id, len(state["call_indexes"])
                )

                emit(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": index,
                                            "id": call_id,
                                            "type": "function",
                                            "function": {
                                                "name": function_state.get("name", ""),
                                                "arguments": function_state.get(
                                                    "arguments", "{}"
                                                ),
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                )
                state["emitted_call_ids"].add(call_id)

    elif event_type == "response.completed":
        response = payload.get("response", {})
        if not isinstance(response, dict):
            response = {}

        reasoning_summary = extract_reasoning_content(response)
        if reasoning_summary:
            emit(
                {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": reasoning_summary,
                            }
                        }
                    ]
                }
            )

        output_items = response.get("output", []) if isinstance(response, dict) else []

        if isinstance(output_items, list):
            for item in output_items:
                if not isinstance(item, dict) or item.get("type") != "function_call":
                    continue

                call_id = item.get("call_id") or item.get("id", "")
                if not call_id or call_id in state["emitted_call_ids"]:
                    continue

                function_state = state["function_calls"].setdefault(
                    call_id,
                    {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}")
                        if isinstance(item.get("arguments"), str)
                        else json.dumps(item.get("arguments", {})),
                    },
                )

                if item.get("name"):
                    function_state["name"] = item.get("name")

                arguments = item.get("arguments", "")
                if isinstance(arguments, str) and arguments:
                    function_state["arguments"] = arguments

                index = state["call_indexes"].setdefault(
                    call_id, len(state["call_indexes"])
                )

                emit(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": index,
                                            "id": call_id,
                                            "type": "function",
                                            "function": {
                                                "name": function_state.get("name", ""),
                                                "arguments": function_state.get(
                                                    "arguments", "{}"
                                                ),
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                )
                state["emitted_call_ids"].add(call_id)

        if not state.get("text_emitted", False):
            text_parts = []
            for item in output_items if isinstance(output_items, list) else []:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for part in item.get("content", []) or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text_parts.append(part.get("text", ""))

            fallback_text = "".join(text_parts)
            if fallback_text:
                emit({"choices": [{"delta": {"content": fallback_text}}]})

        usage = _normalize_responses_usage(response.get("usage"))
        if usage:
            emit({"usage": usage})

        chunks.append("data: [DONE]\n\n")

    return chunks


def responses_output_to_chat_tool_calls(output_items: list[dict]) -> list[dict]:
    tool_calls = []

    for item in output_items or []:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue

        arguments = item.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)

        call_id = item.get("call_id") or item.get("id", "")
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": arguments,
                },
            }
        )

    return tool_calls


def sanitize_responses_metadata(metadata: dict | None) -> dict:
    if not isinstance(metadata, dict):
        return {}

    def _sanitize(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            sanitized = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    continue
                sanitized_item = _sanitize(item)
                if sanitized_item is not None or item is None:
                    sanitized[key] = sanitized_item
            return sanitized
        if isinstance(value, list):
            sanitized = []
            for item in value:
                sanitized_item = _sanitize(item)
                if sanitized_item is not None or item is None:
                    sanitized.append(sanitized_item)
            return sanitized
        return None

    return _sanitize(metadata) or {}


def responses_to_chat_compatible(resp_json: dict) -> dict:
    output = resp_json.get("output", []) if isinstance(resp_json, dict) else []
    content_parts = []

    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                content_parts.append(part.get("text", ""))

    content = "".join(content_parts)
    tool_calls = responses_output_to_chat_tool_calls(output)

    message = {
        "role": "assistant",
        "content": content,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    reasoning_summary = extract_reasoning_content(resp_json)
    if reasoning_summary:
        message["reasoning_content"] = reasoning_summary

    usage = resp_json.get("usage") if isinstance(resp_json, dict) else None
    normalized_usage = None
    if isinstance(usage, dict):
        normalized_usage = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    return {
        "id": resp_json.get("id"),
        "object": "chat.completion",
        "created": resp_json.get("created_at"),
        "model": resp_json.get("model"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        **({"usage": normalized_usage} if normalized_usage else {}),
    }


def chat_to_responses_payload(payload: dict, metadata: Optional[dict], api_config: dict) -> dict:
    responses_payload = {
        "model": payload.get("model"),
        "input": responses_input_from_tool_followups(payload.get("messages", [])),
    }

    if "stream" in payload:
        responses_payload["stream"] = bool(payload.get("stream"))

    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        responses_payload["tools"] = _normalize_tools_for_responses(tools)

    tool_choice = _normalize_tool_choice_for_responses(payload.get("tool_choice"))
    if tool_choice is not None:
        responses_payload["tool_choice"] = tool_choice

    if "parallel_tool_calls" in payload:
        responses_payload["parallel_tool_calls"] = payload.get("parallel_tool_calls")

    max_output_tokens = payload.get("max_completion_tokens", payload.get("max_tokens"))
    if max_output_tokens is not None:
        responses_payload["max_output_tokens"] = max_output_tokens

    reasoning = payload.get("reasoning", {})
    if not isinstance(reasoning, dict):
        reasoning = {}

    explicit_reasoning_summary = None
    if "summary" in reasoning:
        explicit_reasoning_summary = reasoning.get("summary")
    elif payload.get("summary") is not None:
        # support custom_params.summary -> reasoning.summary mapping
        explicit_reasoning_summary = payload.get("summary")

    if payload.get("reasoning_effort"):
        reasoning["effort"] = payload.get("reasoning_effort")

    if is_responses_provider(api_config):
        reasoning["summary"] = normalize_reasoning_summary(
            explicit_reasoning_summary if explicit_reasoning_summary is not None else "auto",
            default="auto",
            source="summary" if explicit_reasoning_summary is not None else "default",
        )

    if reasoning:
        responses_payload["reasoning"] = reasoning

    for key in [
        "temperature",
        "top_p",
        "stop",
        "store",
        "truncation",
        "text",
        "user",
    ]:
        if key in payload:
            responses_payload[key] = payload[key]

    # Only forward explicitly provided request metadata and drop non-JSON values.
    if "metadata" in payload:
        sanitized_metadata = sanitize_responses_metadata(payload.get("metadata"))
        if sanitized_metadata:
            responses_payload["metadata"] = sanitized_metadata

    return responses_payload


def responses_stream_to_chat_streaming_response(stream: aiohttp.StreamReader) -> StreamingResponse:
    async def event_stream():
        buffer = ""
        state = {
            "function_calls": {},
            "call_indexes": {},
            "emitted_call_ids": set(),
            "text_emitted": False,
        }

        async for chunk in stream_chunks_handler(stream):
            if isinstance(chunk, bytes):
                buffer += chunk.decode("utf-8", "replace")
            else:
                buffer += str(chunk)

            while "\n\n" in buffer:
                raw_event, buffer = buffer.split("\n\n", 1)
                if not raw_event.strip():
                    continue

                event_name = None
                data_lines = []
                for line in raw_event.split("\n"):
                    if line.startswith("event:"):
                        event_name = line[len("event:") :].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[len("data:") :].strip())

                if not data_lines:
                    continue

                data_str = "\n".join(data_lines)
                if data_str == "[DONE]":
                    yield "data: [DONE]\n\n"
                    return

                try:
                    payload = json.loads(data_str)
                except Exception:
                    continue

                event_type = event_name or payload.get("type")
                for mapped_chunk in _responses_event_to_chat_chunks(
                    event_type=event_type,
                    payload=payload,
                    state=state,
                ):
                    yield mapped_chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def get_headers_and_cookies(
    request: Request,
    url,
    key=None,
    config=None,
    metadata: Optional[dict] = None,
    user: UserModel = None,
):
    cookies = {}
    headers = {
        "Content-Type": "application/json",
        **(
            {
                "HTTP-Referer": "https://openwebui.com/",
                "X-Title": "Open WebUI",
            }
            if "openrouter.ai" in url
            else {}
        ),
    }

    if ENABLE_FORWARD_USER_INFO_HEADERS and user:
        headers = include_user_info_headers(headers, user)
        if metadata and metadata.get("chat_id"):
            headers["X-OpenWebUI-Chat-Id"] = metadata.get("chat_id")

    token = None
    auth_type = config.get("auth_type")

    if auth_type == "bearer" or auth_type is None:
        # Default to bearer if not specified
        token = f"{key}"
    elif auth_type == "none":
        token = None
    elif auth_type == "session":
        cookies = request.cookies
        token = request.state.token.credentials
    elif auth_type == "system_oauth":
        cookies = request.cookies

        oauth_token = None
        try:
            if request.cookies.get("oauth_session_id", None):
                oauth_token = await request.app.state.oauth_manager.get_oauth_token(
                    user.id,
                    request.cookies.get("oauth_session_id", None),
                )
        except Exception as e:
            log.error(f"Error getting OAuth token: {e}")

        if oauth_token:
            token = f"{oauth_token.get('access_token', '')}"

    elif auth_type in ("azure_ad", "microsoft_entra_id"):
        token = get_microsoft_entra_id_access_token()

    if token:
        headers["Authorization"] = f"Bearer {token}"

    if config.get("headers") and isinstance(config.get("headers"), dict):
        headers = {**headers, **config.get("headers")}

    return headers, cookies


def get_microsoft_entra_id_access_token():
    """
    Get Microsoft Entra ID access token using DefaultAzureCredential for Azure OpenAI.
    Returns the token string or None if authentication fails.
    """
    try:
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        return token_provider()
    except Exception as e:
        log.error(f"Error getting Microsoft Entra ID access token: {e}")
        return None


##########################################
#
# API routes
#
##########################################

router = APIRouter()


@router.get("/config")
async def get_config(request: Request, user=Depends(get_admin_user)):
    return {
        "ENABLE_OPENAI_API": request.app.state.config.ENABLE_OPENAI_API,
        "OPENAI_API_BASE_URLS": request.app.state.config.OPENAI_API_BASE_URLS,
        "OPENAI_API_KEYS": request.app.state.config.OPENAI_API_KEYS,
        "OPENAI_API_CONFIGS": request.app.state.config.OPENAI_API_CONFIGS,
    }


class OpenAIConfigForm(BaseModel):
    ENABLE_OPENAI_API: Optional[bool] = None
    OPENAI_API_BASE_URLS: list[str]
    OPENAI_API_KEYS: list[str]
    OPENAI_API_CONFIGS: dict


@router.post("/config/update")
async def update_config(
    request: Request, form_data: OpenAIConfigForm, user=Depends(get_admin_user)
):
    request.app.state.config.ENABLE_OPENAI_API = form_data.ENABLE_OPENAI_API
    request.app.state.config.OPENAI_API_BASE_URLS = form_data.OPENAI_API_BASE_URLS
    request.app.state.config.OPENAI_API_KEYS = form_data.OPENAI_API_KEYS

    # Check if API KEYS length is same than API URLS length
    if len(request.app.state.config.OPENAI_API_KEYS) != len(
        request.app.state.config.OPENAI_API_BASE_URLS
    ):
        if len(request.app.state.config.OPENAI_API_KEYS) > len(
            request.app.state.config.OPENAI_API_BASE_URLS
        ):
            request.app.state.config.OPENAI_API_KEYS = (
                request.app.state.config.OPENAI_API_KEYS[
                    : len(request.app.state.config.OPENAI_API_BASE_URLS)
                ]
            )
        else:
            request.app.state.config.OPENAI_API_KEYS += [""] * (
                len(request.app.state.config.OPENAI_API_BASE_URLS)
                - len(request.app.state.config.OPENAI_API_KEYS)
            )

    request.app.state.config.OPENAI_API_CONFIGS = form_data.OPENAI_API_CONFIGS

    # Remove the API configs that are not in the API URLS
    keys = list(map(str, range(len(request.app.state.config.OPENAI_API_BASE_URLS))))
    request.app.state.config.OPENAI_API_CONFIGS = {
        key: value
        for key, value in request.app.state.config.OPENAI_API_CONFIGS.items()
        if key in keys
    }

    return {
        "ENABLE_OPENAI_API": request.app.state.config.ENABLE_OPENAI_API,
        "OPENAI_API_BASE_URLS": request.app.state.config.OPENAI_API_BASE_URLS,
        "OPENAI_API_KEYS": request.app.state.config.OPENAI_API_KEYS,
        "OPENAI_API_CONFIGS": request.app.state.config.OPENAI_API_CONFIGS,
    }


@router.post("/audio/speech")
async def speech(request: Request, user=Depends(get_verified_user)):
    idx = None
    try:
        idx = request.app.state.config.OPENAI_API_BASE_URLS.index(
            "https://api.openai.com/v1"
        )

        body = await request.body()
        name = hashlib.sha256(body).hexdigest()

        SPEECH_CACHE_DIR = CACHE_DIR / "audio" / "speech"
        SPEECH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        file_path = SPEECH_CACHE_DIR.joinpath(f"{name}.mp3")
        file_body_path = SPEECH_CACHE_DIR.joinpath(f"{name}.json")

        # Check if the file already exists in the cache
        if file_path.is_file():
            return FileResponse(file_path)

        url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
        key = request.app.state.config.OPENAI_API_KEYS[idx]
        api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
            str(idx),
            request.app.state.config.OPENAI_API_CONFIGS.get(url, {}),  # Legacy support
        )

        headers, cookies = await get_headers_and_cookies(
            request, url, key, api_config, user=user
        )

        r = None
        try:
            r = requests.post(
                url=f"{url}/audio/speech",
                data=body,
                headers=headers,
                cookies=cookies,
                stream=True,
            )

            r.raise_for_status()

            # Save the streaming content to a file
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            with open(file_body_path, "w") as f:
                json.dump(json.loads(body.decode("utf-8")), f)

            # Return the saved file
            return FileResponse(file_path)

        except Exception as e:
            log.exception(e)

            detail = None
            if r is not None:
                try:
                    res = r.json()
                    if "error" in res:
                        detail = f"External: {res['error']}"
                except Exception:
                    detail = f"External: {e}"

            raise HTTPException(
                status_code=r.status_code if r else 500,
                detail=detail if detail else "Open WebUI: Server Connection Error",
            )

    except ValueError:
        raise HTTPException(status_code=401, detail=ERROR_MESSAGES.OPENAI_NOT_FOUND)


async def get_all_models_responses(request: Request, user: UserModel) -> list:
    if not request.app.state.config.ENABLE_OPENAI_API:
        return []

    # Check if API KEYS length is same than API URLS length
    num_urls = len(request.app.state.config.OPENAI_API_BASE_URLS)
    num_keys = len(request.app.state.config.OPENAI_API_KEYS)

    if num_keys != num_urls:
        # if there are more keys than urls, remove the extra keys
        if num_keys > num_urls:
            new_keys = request.app.state.config.OPENAI_API_KEYS[:num_urls]
            request.app.state.config.OPENAI_API_KEYS = new_keys
        # if there are more urls than keys, add empty keys
        else:
            request.app.state.config.OPENAI_API_KEYS += [""] * (num_urls - num_keys)

    request_tasks = []
    for idx, url in enumerate(request.app.state.config.OPENAI_API_BASE_URLS):
        if (str(idx) not in request.app.state.config.OPENAI_API_CONFIGS) and (
            url not in request.app.state.config.OPENAI_API_CONFIGS  # Legacy support
        ):
            request_tasks.append(
                send_get_request(
                    f"{url}/models",
                    request.app.state.config.OPENAI_API_KEYS[idx],
                    user=user,
                )
            )
        else:
            api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
                str(idx),
                request.app.state.config.OPENAI_API_CONFIGS.get(
                    url, {}
                ),  # Legacy support
            )

            enable = api_config.get("enable", True)
            model_ids = api_config.get("model_ids", [])

            if enable:
                if len(model_ids) == 0:
                    request_tasks.append(
                        send_get_request(
                            f"{url}/models",
                            request.app.state.config.OPENAI_API_KEYS[idx],
                            user=user,
                        )
                    )
                else:
                    model_list = {
                        "object": "list",
                        "data": [
                            {
                                "id": model_id,
                                "name": model_id,
                                "owned_by": "openai",
                                "openai": {"id": model_id},
                                "urlIdx": idx,
                            }
                            for model_id in model_ids
                        ],
                    }

                    request_tasks.append(
                        asyncio.ensure_future(asyncio.sleep(0, model_list))
                    )
            else:
                request_tasks.append(asyncio.ensure_future(asyncio.sleep(0, None)))

    responses = await asyncio.gather(*request_tasks)

    for idx, response in enumerate(responses):
        if response:
            url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
            api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
                str(idx),
                request.app.state.config.OPENAI_API_CONFIGS.get(
                    url, {}
                ),  # Legacy support
            )

            connection_type = api_config.get("connection_type", "external")
            prefix_id = api_config.get("prefix_id", None)
            tags = api_config.get("tags", [])

            model_list = (
                response if isinstance(response, list) else response.get("data", [])
            )
            if not isinstance(model_list, list):
                # Catch non-list responses
                model_list = []

            for model in model_list:
                # Remove name key if its value is None #16689
                if "name" in model and model["name"] is None:
                    del model["name"]

                if prefix_id:
                    model["id"] = (
                        f"{prefix_id}.{model.get('id', model.get('name', ''))}"
                    )

                if tags:
                    model["tags"] = tags

                if connection_type:
                    model["connection_type"] = connection_type

    log.debug(f"get_all_models:responses() {responses}")
    return responses


async def get_filtered_models(models, user):
    # Filter models based on user access control
    filtered_models = []
    for model in models.get("data", []):
        model_info = Models.get_model_by_id(model["id"])
        if model_info:
            if user.id == model_info.user_id or has_access(
                user.id, type="read", access_control=model_info.access_control
            ):
                filtered_models.append(model)
    return filtered_models


@cached(
    ttl=MODELS_CACHE_TTL,
    key=lambda _, user: f"openai_all_models_{user.id}" if user else "openai_all_models",
)
async def get_all_models(request: Request, user: UserModel) -> dict[str, list]:
    log.info("get_all_models()")

    if not request.app.state.config.ENABLE_OPENAI_API:
        return {"data": []}

    responses = await get_all_models_responses(request, user=user)

    def extract_data(response):
        if response and "data" in response:
            return response["data"]
        if isinstance(response, list):
            return response
        return None

    def is_supported_openai_models(model_id):
        if any(
            name in model_id
            for name in [
                "babbage",
                "dall-e",
                "davinci",
                "embedding",
                "tts",
                "whisper",
            ]
        ):
            return False
        return True

    def get_merged_models(model_lists):
        log.debug(f"merge_models_lists {model_lists}")
        models = {}

        for idx, model_list in enumerate(model_lists):
            if model_list is not None and "error" not in model_list:
                for model in model_list:
                    model_id = model.get("id") or model.get("name")

                    if (
                        "api.openai.com"
                        in request.app.state.config.OPENAI_API_BASE_URLS[idx]
                        and not is_supported_openai_models(model_id)
                    ):
                        # Skip unwanted OpenAI models
                        continue

                    if model_id and model_id not in models:
                        models[model_id] = {
                            **model,
                            "name": model.get("name", model_id),
                            "owned_by": "openai",
                            "openai": model,
                            "connection_type": model.get("connection_type", "external"),
                            "urlIdx": idx,
                        }

        return models

    models = get_merged_models(map(extract_data, responses))
    log.debug(f"models: {models}")

    request.app.state.OPENAI_MODELS = models
    return {"data": list(models.values())}


@router.get("/models")
@router.get("/models/{url_idx}")
async def get_models(
    request: Request, url_idx: Optional[int] = None, user=Depends(get_verified_user)
):
    models = {
        "data": [],
    }

    if url_idx is None:
        models = await get_all_models(request, user=user)
    else:
        url = request.app.state.config.OPENAI_API_BASE_URLS[url_idx]
        key = request.app.state.config.OPENAI_API_KEYS[url_idx]

        api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
            str(url_idx),
            request.app.state.config.OPENAI_API_CONFIGS.get(url, {}),  # Legacy support
        )

        r = None
        async with aiohttp.ClientSession(
            trust_env=True,
            timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST),
        ) as session:
            try:
                headers, cookies = await get_headers_and_cookies(
                    request, url, key, api_config, user=user
                )

                if get_provider_type(api_config) == "azure_openai":
                    models = {
                        "data": api_config.get("model_ids", []) or [],
                        "object": "list",
                    }
                else:
                    async with session.get(
                        f"{url}/models",
                        headers=headers,
                        cookies=cookies,
                        ssl=AIOHTTP_CLIENT_SESSION_SSL,
                    ) as r:
                        if r.status != 200:
                            # Extract response error details if available
                            error_detail = f"HTTP Error: {r.status}"
                            res = await r.json()
                            if "error" in res:
                                error_detail = f"External Error: {res['error']}"
                            raise Exception(error_detail)

                        response_data = await r.json()

                        # Check if we're calling OpenAI API based on the URL
                        if "api.openai.com" in url:
                            # Filter models according to the specified conditions
                            response_data["data"] = [
                                model
                                for model in response_data.get("data", [])
                                if not any(
                                    name in model["id"]
                                    for name in [
                                        "babbage",
                                        "dall-e",
                                        "davinci",
                                        "embedding",
                                        "tts",
                                        "whisper",
                                    ]
                                )
                            ]

                        models = response_data
            except aiohttp.ClientError as e:
                # ClientError covers all aiohttp requests issues
                log.exception(f"Client error: {str(e)}")
                raise HTTPException(
                    status_code=500, detail="Open WebUI: Server Connection Error"
                )
            except Exception as e:
                log.exception(f"Unexpected error: {e}")
                error_detail = f"Unexpected error: {str(e)}"
                raise HTTPException(status_code=500, detail=error_detail)

    if user.role == "user" and not BYPASS_MODEL_ACCESS_CONTROL:
        models["data"] = await get_filtered_models(models, user)

    return models


class ConnectionVerificationForm(BaseModel):
    url: str
    key: str

    config: Optional[dict] = None


@router.post("/verify")
async def verify_connection(
    request: Request,
    form_data: ConnectionVerificationForm,
    user=Depends(get_admin_user),
):
    url = form_data.url
    key = form_data.key

    api_config = form_data.config or {}

    async with aiohttp.ClientSession(
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST),
    ) as session:
        try:
            headers, cookies = await get_headers_and_cookies(
                request, url, key, api_config, user=user
            )

            if get_provider_type(api_config) == "azure_openai":
                # Only set api-key header if not using Azure Entra ID authentication
                auth_type = api_config.get("auth_type", "bearer")
                if auth_type not in ("azure_ad", "microsoft_entra_id"):
                    headers["api-key"] = key

                api_version = api_config.get("api_version", "") or "2023-03-15-preview"
                async with session.get(
                    url=f"{url}/openai/models?api-version={api_version}",
                    headers=headers,
                    cookies=cookies,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                ) as r:
                    try:
                        response_data = await r.json()
                    except Exception:
                        response_data = await r.text()

                    if r.status != 200:
                        if isinstance(response_data, (dict, list)):
                            return JSONResponse(
                                status_code=r.status, content=response_data
                            )
                        else:
                            return PlainTextResponse(
                                status_code=r.status, content=response_data
                            )

                    return response_data
            else:
                async with session.get(
                    f"{url}/models",
                    headers=headers,
                    cookies=cookies,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                ) as r:
                    try:
                        response_data = await r.json()
                    except Exception:
                        response_data = await r.text()

                    if r.status != 200:
                        if isinstance(response_data, (dict, list)):
                            return JSONResponse(
                                status_code=r.status, content=response_data
                            )
                        else:
                            return PlainTextResponse(
                                status_code=r.status, content=response_data
                            )

                    return response_data

        except aiohttp.ClientError as e:
            # ClientError covers all aiohttp requests issues
            log.exception(f"Client error: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Open WebUI: Server Connection Error"
            )
        except Exception as e:
            log.exception(f"Unexpected error: {e}")
            raise HTTPException(
                status_code=500, detail="Open WebUI: Server Connection Error"
            )


def get_azure_allowed_params(api_version: str) -> set[str]:
    allowed_params = {
        "messages",
        "temperature",
        "role",
        "content",
        "contentPart",
        "contentPartImage",
        "enhancements",
        "dataSources",
        "n",
        "stream",
        "stop",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "function_call",
        "functions",
        "tools",
        "tool_choice",
        "top_p",
        "log_probs",
        "top_logprobs",
        "response_format",
        "seed",
        "max_completion_tokens",
        "reasoning_effort",
    }

    try:
        if api_version >= "2024-09-01-preview":
            allowed_params.add("stream_options")
    except ValueError:
        log.debug(
            f"Invalid API version {api_version} for Azure OpenAI. Defaulting to allowed parameters."
        )

    return allowed_params


def is_openai_reasoning_model(model: str) -> bool:
    return model.lower().startswith(("o1", "o3", "o4", "gpt-5"))


def convert_to_azure_payload(url, payload: dict, api_version: str):
    model = payload.get("model", "")

    # Filter allowed parameters based on Azure OpenAI API
    allowed_params = get_azure_allowed_params(api_version)

    # Special handling for o-series models
    if is_openai_reasoning_model(model):
        # Convert max_tokens to max_completion_tokens for o-series models
        if "max_tokens" in payload:
            payload["max_completion_tokens"] = payload["max_tokens"]
            del payload["max_tokens"]

        # Remove temperature if not 1 for o-series models
        if "temperature" in payload and payload["temperature"] != 1:
            log.debug(
                f"Removing temperature parameter for o-series model {model} as only default value (1) is supported"
            )
            del payload["temperature"]

    # Filter out unsupported parameters
    payload = {k: v for k, v in payload.items() if k in allowed_params}

    url = f"{url}/openai/deployments/{model}"
    return url, payload


@router.post("/chat/completions")
async def generate_chat_completion(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
    bypass_filter: Optional[bool] = False,
):
    if BYPASS_MODEL_ACCESS_CONTROL:
        bypass_filter = True

    idx = 0

    payload = {**form_data}
    metadata = payload.pop("metadata", None)

    model_id = form_data.get("model")
    model_info = Models.get_model_by_id(model_id)

    # Check model info and override the payload
    if model_info:
        if model_info.base_model_id:
            base_model_id = (
                request.base_model_id
                if hasattr(request, "base_model_id")
                else model_info.base_model_id
            )  # Use request's base_model_id if available
            payload["model"] = base_model_id
            model_id = base_model_id

        params = model_info.params.model_dump()

        if params:
            system = params.pop("system", None)

            payload = apply_model_params_to_body_openai(params, payload)
            payload = apply_system_prompt_to_body(system, payload, metadata, user)

        # Check if user has access to the model
        if not bypass_filter and user.role == "user":
            if not (
                user.id == model_info.user_id
                or has_access(
                    user.id, type="read", access_control=model_info.access_control
                )
            ):
                raise HTTPException(
                    status_code=403,
                    detail="Model not found",
                )
    elif not bypass_filter:
        if user.role != "admin":
            raise HTTPException(
                status_code=403,
                detail="Model not found",
            )

    await get_all_models(request, user=user)
    model = request.app.state.OPENAI_MODELS.get(model_id)
    if model:
        idx = model["urlIdx"]
    else:
        raise HTTPException(
            status_code=404,
            detail="Model not found",
        )

    # Get the API config for the model
    api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
        str(idx),
        request.app.state.config.OPENAI_API_CONFIGS.get(
            request.app.state.config.OPENAI_API_BASE_URLS[idx], {}
        ),  # Legacy support
    )

    prefix_id = api_config.get("prefix_id", None)
    if prefix_id:
        payload["model"] = payload["model"].replace(f"{prefix_id}.", "")

    # Add user info to the payload if the model is a pipeline
    if "pipeline" in model and model.get("pipeline"):
        payload["user"] = {
            "name": user.name,
            "id": user.id,
            "email": user.email,
            "role": user.role,
        }

    url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
    key = request.app.state.config.OPENAI_API_KEYS[idx]

    provider_type = get_provider_type(api_config)

    # Check if model is a reasoning model that needs special handling
    if is_openai_reasoning_model(payload["model"]) and not is_responses_provider(api_config):
        payload = openai_reasoning_model_handler(payload)
    elif "api.openai.com" not in url and not is_responses_provider(api_config):
        # Remove "max_completion_tokens" from the payload for backward compatibility
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload["max_completion_tokens"]
            del payload["max_completion_tokens"]

    if "max_tokens" in payload and "max_completion_tokens" in payload:
        del payload["max_tokens"]

    # Convert the modified body back to JSON
    if "logit_bias" in payload and payload["logit_bias"]:
        logit_bias = convert_logit_bias_input_to_json(payload["logit_bias"])

        if logit_bias:
            payload["logit_bias"] = json.loads(logit_bias)

    headers, cookies = await get_headers_and_cookies(
        request, url, key, api_config, metadata, user=user
    )

    if provider_type == "azure_openai":
        api_version = api_config.get("api_version", "2023-03-15-preview")
        request_url, payload = convert_to_azure_payload(url, payload, api_version)

        # Only set api-key header if not using Azure Entra ID authentication
        auth_type = api_config.get("auth_type", "bearer")
        if auth_type not in ("azure_ad", "microsoft_entra_id"):
            headers["api-key"] = key

        headers["api-version"] = api_version
        request_url = f"{request_url}/chat/completions?api-version={api_version}"
    elif provider_type == "openai_responses":
        payload = chat_to_responses_payload(payload, metadata, api_config)
        request_url = f"{url}/responses"
    else:
        request_url = f"{url}/chat/completions"

    payload = json.dumps(payload)

    r = None
    session = None
    streaming = False
    response = None

    try:
        session = aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
        )

        r = await session.request(
            method="POST",
            url=request_url,
            data=payload,
            headers=headers,
            cookies=cookies,
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
        )

        # Check if response is SSE
        if "text/event-stream" in r.headers.get("Content-Type", ""):
            streaming = True
            if provider_type == "openai_responses":
                return responses_stream_to_chat_streaming_response(r.content)

            return StreamingResponse(
                stream_chunks_handler(r.content),
                status_code=r.status,
                headers=dict(r.headers),
                background=BackgroundTask(
                    cleanup_response, response=r, session=session
                ),
            )
        else:
            try:
                response = await r.json()
            except Exception as e:
                log.error(e)
                response = await r.text()

            if r.status >= 400:
                if isinstance(response, (dict, list)):
                    return JSONResponse(status_code=r.status, content=response)
                else:
                    return PlainTextResponse(status_code=r.status, content=response)

            if provider_type == "openai_responses" and isinstance(response, dict):
                return responses_to_chat_compatible(response)

            return response
    except Exception as e:
        log.exception(e)

        raise HTTPException(
            status_code=r.status if r else 500,
            detail="Open WebUI: Server Connection Error",
        )
    finally:
        if not streaming:
            await cleanup_response(r, session)


async def embeddings(request: Request, form_data: dict, user):
    """
    Calls the embeddings endpoint for OpenAI-compatible providers.

    Args:
        request (Request): The FastAPI request context.
        form_data (dict): OpenAI-compatible embeddings payload.
        user (UserModel): The authenticated user.

    Returns:
        dict: OpenAI-compatible embeddings response.
    """
    idx = 0
    # Prepare payload/body
    body = json.dumps(form_data)
    # Find correct backend url/key based on model
    await get_all_models(request, user=user)
    model_id = form_data.get("model")
    models = request.app.state.OPENAI_MODELS
    if model_id in models:
        idx = models[model_id]["urlIdx"]

    url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
    key = request.app.state.config.OPENAI_API_KEYS[idx]
    api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
        str(idx),
        request.app.state.config.OPENAI_API_CONFIGS.get(url, {}),  # Legacy support
    )

    r = None
    session = None
    streaming = False

    headers, cookies = await get_headers_and_cookies(
        request, url, key, api_config, user=user
    )
    try:
        session = aiohttp.ClientSession(trust_env=True)
        r = await session.request(
            method="POST",
            url=f"{url}/embeddings",
            data=body,
            headers=headers,
            cookies=cookies,
        )

        if "text/event-stream" in r.headers.get("Content-Type", ""):
            streaming = True
            return StreamingResponse(
                r.content,
                status_code=r.status,
                headers=dict(r.headers),
                background=BackgroundTask(
                    cleanup_response, response=r, session=session
                ),
            )
        else:
            try:
                response_data = await r.json()
            except Exception:
                response_data = await r.text()

            if r.status >= 400:
                if isinstance(response_data, (dict, list)):
                    return JSONResponse(status_code=r.status, content=response_data)
                else:
                    return PlainTextResponse(
                        status_code=r.status, content=response_data
                    )

            return response_data
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=r.status if r else 500,
            detail="Open WebUI: Server Connection Error",
        )
    finally:
        if not streaming:
            await cleanup_response(r, session)


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(path: str, request: Request, user=Depends(get_verified_user)):
    """
    Deprecated: proxy all requests to OpenAI API
    """

    body = await request.body()

    idx = 0
    url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
    key = request.app.state.config.OPENAI_API_KEYS[idx]
    api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
        str(idx),
        request.app.state.config.OPENAI_API_CONFIGS.get(
            request.app.state.config.OPENAI_API_BASE_URLS[idx], {}
        ),  # Legacy support
    )

    r = None
    session = None
    streaming = False

    try:
        headers, cookies = await get_headers_and_cookies(
            request, url, key, api_config, user=user
        )

        if get_provider_type(api_config) == "azure_openai":
            api_version = api_config.get("api_version", "2023-03-15-preview")

            # Only set api-key header if not using Azure Entra ID authentication
            auth_type = api_config.get("auth_type", "bearer")
            if auth_type not in ("azure_ad", "microsoft_entra_id"):
                headers["api-key"] = key

            headers["api-version"] = api_version

            payload = json.loads(body)
            url, payload = convert_to_azure_payload(url, payload, api_version)
            body = json.dumps(payload).encode()

            request_url = f"{url}/{path}?api-version={api_version}"
        else:
            request_url = f"{url}/{path}"

        session = aiohttp.ClientSession(trust_env=True)
        r = await session.request(
            method=request.method,
            url=request_url,
            data=body,
            headers=headers,
            cookies=cookies,
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
        )

        # Check if response is SSE
        if "text/event-stream" in r.headers.get("Content-Type", ""):
            streaming = True
            return StreamingResponse(
                r.content,
                status_code=r.status,
                headers=dict(r.headers),
                background=BackgroundTask(
                    cleanup_response, response=r, session=session
                ),
            )
        else:
            try:
                response_data = await r.json()
            except Exception:
                response_data = await r.text()

            if r.status >= 400:
                if isinstance(response_data, (dict, list)):
                    return JSONResponse(status_code=r.status, content=response_data)
                else:
                    return PlainTextResponse(
                        status_code=r.status, content=response_data
                    )

            return response_data

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=r.status if r else 500,
            detail="Open WebUI: Server Connection Error",
        )
    finally:
        if not streaming:
            await cleanup_response(r, session)
