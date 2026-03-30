import hashlib
import json
import logging
import uuid
from typing import Any, Optional

from open_webui.models.chats import Chats


log = logging.getLogger(__name__)

INTERNAL_FORM_KEYS = {
    "model_item",
    "endpointKind",
    "endpoint_kind",
    "_responses_upstream_payload",
    "chat_id",
    "id",
    "parent_id",
    "parent_message",
    "session_id",
    "background_tasks",
    "filter_ids",
    "tool_ids",
    "files",
    "features",
    "variables",
}


def resolve_endpoint_kind(
    explicit_endpoint: Optional[str] = None,
    provider_type: Optional[str] = None,
    provider_hint: Optional[dict] = None,
) -> str:
    if explicit_endpoint in {"chat_completions", "responses"}:
        return explicit_endpoint

    resolved_provider = provider_type
    if not isinstance(resolved_provider, str) and isinstance(provider_hint, dict):
        resolved_provider = provider_hint.get("provider_type") or (
            provider_hint.get("info", {}).get("provider_type")
            if isinstance(provider_hint.get("info"), dict)
            else None
        )

    return "responses" if resolved_provider == "openai_responses" else "chat_completions"


def provider_type_from_model_id(
    model_id: Optional[str],
    models: Optional[dict],
    openai_models: Optional[dict] = None,
) -> Optional[str]:
    if isinstance(openai_models, dict) and isinstance(model_id, str):
        openai_model = openai_models.get(model_id)
        if isinstance(openai_model, dict):
            provider_type = openai_model.get("provider_type")
            if isinstance(provider_type, str):
                return provider_type

    if isinstance(models, dict) and isinstance(model_id, str):
        model = models.get(model_id)
        if isinstance(model, dict):
            provider_type = model.get("provider_type")
            if isinstance(provider_type, str):
                return provider_type

            info = model.get("info")
            if isinstance(info, dict):
                info_provider_type = info.get("provider_type")
                if isinstance(info_provider_type, str):
                    return info_provider_type

    return None


def chat_messages_to_responses_input(messages: list[dict]) -> list[dict]:
    input_items: list[dict] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": content if isinstance(content, str) else json.dumps(content),
                }
            )
            continue

        tool_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function", {})
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

            if content:
                text_content = content if isinstance(content, str) else str(content)
                input_items.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text_content}],
                    }
                )
            continue

        if isinstance(content, str):
            input_items.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            normalized_parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    normalized_parts.append(
                        {
                            "type": "output_text" if role == "assistant" else "input_text",
                            "text": part.get("text", ""),
                        }
                    )
                elif part_type == "image_url":
                    image_url = part.get("image_url", {})
                    normalized_parts.append(
                        {
                            "type": "input_image",
                            "image_url": image_url.get("url", "")
                            if isinstance(image_url, dict)
                            else image_url,
                        }
                    )
                elif part_type in {"input_text", "output_text", "input_image", "input_file"}:
                    normalized_parts.append(part)
            if normalized_parts:
                input_items.append({"role": role, "content": normalized_parts})

    return input_items


def normalize_tools_for_responses(tools: list[dict]) -> list[dict]:
    normalized_tools: list[dict] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue

        if tool.get("type") != "function":
            normalized_tools.append(tool)
            continue

        function_spec = tool.get("function")
        if isinstance(function_spec, dict):
            normalized_tool = {
                "type": "function",
                "name": function_spec.get("name", ""),
                "parameters": (
                    function_spec.get("parameters")
                    if isinstance(function_spec.get("parameters"), dict)
                    else {}
                ),
            }
            if isinstance(function_spec.get("description"), str):
                normalized_tool["description"] = function_spec.get("description")
            if isinstance(function_spec.get("strict"), bool):
                normalized_tool["strict"] = function_spec.get("strict")
            normalized_tools.append(normalized_tool)
            continue

        normalized_tools.append(tool)

    return normalized_tools


def extract_text_from_responses_payload(response: dict) -> str:
    if not isinstance(response, dict):
        return ""

    output = response.get("output")
    if not isinstance(output, list):
        fallback = response.get("output_text")
        return fallback if isinstance(fallback, str) else ""

    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "output_text":
                continue
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)

    return "".join(chunks)


def extract_reasoning_text_from_responses_payload(response: dict) -> str:
    if not isinstance(response, dict):
        return ""

    output = response.get("output")
    if not isinstance(output, list):
        return ""

    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "reasoning":
            continue
        summary = item.get("summary")
        if isinstance(summary, str):
            chunks.append(summary)
            continue
        if not isinstance(summary, list):
            continue
        for part in summary:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)

    return "\n".join([chunk for chunk in chunks if chunk])


def extract_assistant_content_from_completion_response(response: dict) -> str:
    if not isinstance(response, dict):
        return ""

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and isinstance(p.get("text"), str)
                ]
            )

    return extract_text_from_responses_payload(response)


def build_chat_compatible_response_from_responses_payload(response: dict) -> dict:
    if not isinstance(response, dict):
        return response
    if isinstance(response.get("choices"), list):
        return response

    content = extract_text_from_responses_payload(response)
    reasoning = extract_reasoning_text_from_responses_payload(response)

    message = {
        "role": "assistant",
        "content": content,
    }
    if reasoning:
        message["reasoning_content"] = reasoning

    return {
        **response,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
    }


def build_upstream_payload(
    form_data: dict,
    endpoint_kind: str,
    *,
    metadata: Optional[dict] = None,
    base_payload: Optional[dict] = None,
    strip_internal_keys: bool = False,
    include_endpoint_kind: bool = True,
) -> dict:
    payload = {**(base_payload if isinstance(base_payload, dict) else form_data)}

    if endpoint_kind == "responses":
        if isinstance(form_data.get("messages"), list):
            payload["input"] = chat_messages_to_responses_input(
                form_data.get("messages", [])
            )
        if isinstance(form_data.get("tools"), list):
            payload["tools"] = normalize_tools_for_responses(
                form_data.get("tools", [])
            )
        if "tool_choice" in form_data:
            payload["tool_choice"] = form_data.get("tool_choice")
        if "parallel_tool_calls" in form_data:
            payload["parallel_tool_calls"] = form_data.get("parallel_tool_calls")
        payload.pop("messages", None)

    if strip_internal_keys:
        for key in INTERNAL_FORM_KEYS:
            payload.pop(key, None)

    if isinstance(metadata, dict):
        payload["metadata"] = metadata

    if include_endpoint_kind:
        payload["endpoint_kind"] = endpoint_kind

    return payload


def _mask_prompt_cache_key(prompt_cache_key: Optional[str]) -> str:
    if not isinstance(prompt_cache_key, str) or not prompt_cache_key:
        return "<none>"
    if len(prompt_cache_key) <= 8:
        return prompt_cache_key
    return f"{prompt_cache_key[:8]}..."


def resolve_prompt_cache_key_for_completion_request(
    payload: dict, metadata: Optional[dict], user: Any
) -> Optional[str]:
    explicit_key = payload.get("prompt_cache_key")
    if isinstance(explicit_key, str) and explicit_key.strip():
        resolved_key = explicit_key.strip()
        log.debug(
            "Completion prompt_cache_key source=explicit key=%s",
            _mask_prompt_cache_key(resolved_key),
        )
        return resolved_key

    chat_id = metadata.get("chat_id") if isinstance(metadata, dict) else None
    if not isinstance(chat_id, str) or not chat_id:
        log.debug("Completion prompt_cache_key source=none key=<none>")
        return None

    if chat_id.startswith("local:"):
        derived_key = f"pc:v1:local:{hashlib.sha256(chat_id.encode()).hexdigest()[:16]}"
        log.debug(
            "Completion prompt_cache_key source=local_derived chat_id=%s key=%s",
            chat_id,
            _mask_prompt_cache_key(derived_key),
        )
        return derived_key

    chat = Chats.get_chat_by_id_and_user_id(chat_id, user.id)
    if chat is None and getattr(user, "role", None) == "admin":
        chat = Chats.get_chat_by_id(chat_id)

    if chat is None:
        log.debug(
            "Completion prompt_cache_key source=none chat_id=%s key=<none>", chat_id
        )
        return None

    chat_meta = chat.meta if isinstance(chat.meta, dict) else {}
    chat_meta_key = chat_meta.get("prompt_cache_key")
    if isinstance(chat_meta_key, str) and chat_meta_key.strip():
        resolved_key = chat_meta_key.strip()
        log.debug(
            "Completion prompt_cache_key source=chat_meta chat_id=%s key=%s",
            chat_id,
            _mask_prompt_cache_key(resolved_key),
        )
        return resolved_key

    generated_key = f"pc:v1:{uuid.uuid4().hex}"
    updated_chat = Chats.update_chat_metadata_by_id(
        chat_id,
        {
            "prompt_cache_key": generated_key,
            "prompt_cache_key_version": "v1",
        },
    )

    log.debug(
        "Completion prompt_cache_key source=generated_persisted chat_id=%s persisted=%s key=%s",
        chat_id,
        bool(updated_chat),
        _mask_prompt_cache_key(generated_key),
    )
    return generated_key


def apply_prompt_cache_policy(
    provider_type: str,
    endpoint_kind: str,
    payload: dict,
    metadata: Optional[dict],
    user: Any,
    retention_mode: str = "force_24h",
) -> None:
    applies_to_responses = (
        endpoint_kind == "responses" and provider_type == "openai_responses"
    )
    applies_to_chat_completions = (
        endpoint_kind == "chat_completions" and provider_type == "openai"
    )
    if not (applies_to_responses or applies_to_chat_completions):
        return

    resolved_prompt_cache_key = resolve_prompt_cache_key_for_completion_request(
        payload, metadata, user
    )
    explicit_prompt_cache_key = payload.get("prompt_cache_key")
    has_explicit_prompt_cache_key = (
        isinstance(explicit_prompt_cache_key, str)
        and bool(explicit_prompt_cache_key.strip())
    )
    if resolved_prompt_cache_key and not has_explicit_prompt_cache_key:
        payload["prompt_cache_key"] = resolved_prompt_cache_key
