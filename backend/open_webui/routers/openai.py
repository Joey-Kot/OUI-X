import asyncio
import hashlib
import json
import logging
from typing import Any, Optional

import aiohttp
from aiocache import cached
import requests

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
from open_webui.models.chats import Chats
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
    apply_model_params_as_defaults_openai,
    apply_model_params_to_body_openai,
)
from open_webui.utils.completion_adapter import (
    apply_prompt_cache_policy,
    resolve_prompt_cache_key_for_completion_request as adapter_resolve_prompt_cache_key,
)
from open_webui.utils.misc import (
    deep_update,
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


CHAT_KNOWN_NULLABLE_PARAM_KEYS = {
    "temperature",
    "top_p",
    "stop",
    "reasoning_effort",
    "verbosity",
    "max_tokens",
    "max_completion_tokens",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "logit_bias",
}


RESPONSES_KNOWN_NULLABLE_PARAM_KEYS = {
    "temperature",
    "top_p",
    "stop",
    "reasoning_effort",
    "summary",
    "verbosity",
    "response_format",
    "reasoning",
    "text",
    "max_output_tokens",
    "max_tokens",
    "max_completion_tokens",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "logit_bias",
}


def _remove_known_nulls(payload: dict, known_keys: set[str]) -> dict:
    cleaned = {**payload}
    for key in known_keys:
        if key in cleaned and cleaned[key] is None:
            del cleaned[key]
    return cleaned


def _normalize_chat_completions_payload_known_params(payload: dict) -> dict:
    """
    Normalize known Chat Completions params only.
    Unknown fields are preserved as-is.
    """
    if not isinstance(payload, dict):
        return payload
    return _remove_known_nulls(payload, CHAT_KNOWN_NULLABLE_PARAM_KEYS)


def _normalize_responses_payload_known_params(payload: dict) -> dict:
    """
    Normalize /responses payload parameter semantics:
    - Treat explicit nulls as unset so model defaults can apply.
    - Map legacy known aliases to Responses schema.
    - Normalize max token aliases to max_output_tokens.
    Unknown fields are preserved as-is.
    """
    if not isinstance(payload, dict):
        return payload

    sanitized = _remove_known_nulls(payload, RESPONSES_KNOWN_NULLABLE_PARAM_KEYS)

    reasoning = sanitized.get("reasoning")
    if isinstance(reasoning, dict):
        reasoning_sanitized = {**reasoning}
        if reasoning_sanitized.get("effort") is None:
            reasoning_sanitized.pop("effort", None)
        if reasoning_sanitized.get("summary") is None:
            reasoning_sanitized.pop("summary", None)

        if not reasoning_sanitized:
            sanitized.pop("reasoning", None)
        else:
            sanitized["reasoning"] = reasoning_sanitized

    # Map legacy alias keys when the canonical nested keys are missing.
    if "reasoning_effort" in sanitized:
        reasoning_obj = sanitized.get("reasoning")
        if reasoning_obj is None:
            reasoning_obj = {}
        if isinstance(reasoning_obj, dict) and reasoning_obj.get("effort") is None:
            reasoning_obj["effort"] = sanitized["reasoning_effort"]
        if isinstance(reasoning_obj, dict) and reasoning_obj:
            sanitized["reasoning"] = reasoning_obj

    if "summary" in sanitized:
        reasoning_obj = sanitized.get("reasoning")
        if reasoning_obj is None:
            reasoning_obj = {}
        if isinstance(reasoning_obj, dict) and reasoning_obj.get("summary") is None:
            reasoning_obj["summary"] = sanitized["summary"]
        if isinstance(reasoning_obj, dict) and reasoning_obj:
            sanitized["reasoning"] = reasoning_obj

    if "verbosity" in sanitized:
        text_obj = sanitized.get("text")
        if text_obj is None:
            text_obj = {}
        if isinstance(text_obj, dict) and text_obj.get("verbosity") is None:
            text_obj["verbosity"] = sanitized["verbosity"]
        if isinstance(text_obj, dict) and text_obj:
            sanitized["text"] = text_obj

    if "response_format" in sanitized:
        text_obj = sanitized.get("text")
        if text_obj is None:
            text_obj = {}
        response_format = sanitized.get("response_format")
        if (
            isinstance(text_obj, dict)
            and isinstance(response_format, dict)
            and text_obj.get("format") is None
        ):
            text_obj["format"] = response_format
        if isinstance(text_obj, dict) and text_obj:
            sanitized["text"] = text_obj

    # Legacy aliases are accepted as input but removed from outgoing payload.
    sanitized.pop("reasoning_effort", None)
    sanitized.pop("summary", None)
    sanitized.pop("verbosity", None)
    sanitized.pop("response_format", None)

    if "max_output_tokens" not in sanitized:
        for alias in ("max_completion_tokens", "max_tokens"):
            if alias in sanitized and sanitized[alias] is not None:
                sanitized["max_output_tokens"] = sanitized[alias]
                break

    # /responses uses max_output_tokens; remove legacy aliases to avoid ambiguity.
    sanitized.pop("max_completion_tokens", None)
    sanitized.pop("max_tokens", None)

    return sanitized


def _normalize_payload_known_params_for_endpoint(payload: dict, endpoint: str) -> dict:
    if endpoint == "responses":
        return _normalize_responses_payload_known_params(payload)
    return _normalize_chat_completions_payload_known_params(payload)


def _is_empty_default_param_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (dict, list, tuple, set)) and len(value) == 0:
        return True
    return False


def _sanitize_model_params_for_defaults(raw_params: dict) -> dict:
    if not isinstance(raw_params, dict):
        return {}

    sanitized: dict = {}
    for key, value in raw_params.items():
        if key == "custom_params":
            if not isinstance(value, dict):
                continue

            custom_params = {
                custom_key: custom_value
                for custom_key, custom_value in value.items()
                if not _is_empty_default_param_value(custom_value)
            }
            if custom_params:
                sanitized[key] = custom_params
            continue

        if _is_empty_default_param_value(value):
            continue
        sanitized[key] = value

    return sanitized


def _extract_model_params(model_info: Optional[Any]) -> dict:
    if model_info and getattr(model_info, "params", None):
        return model_info.params.model_dump()
    return {}


def _build_payload_with_layered_model_defaults(
    *,
    payload: dict,
    endpoint: str,
    metadata: Optional[dict],
    user: UserModel,
    custom_model_info: Optional[Any],
    base_model_info: Optional[Any],
) -> dict:
    """
    Build final upstream payload defaults in one place with precedence:
    request > custom model params > base model params.
    Empty values in all layers are treated as unset to avoid accidental override.
    """
    merged_payload = {**payload}

    base_model_params = _sanitize_model_params_for_defaults(
        _extract_model_params(base_model_info)
    )
    custom_model_params = _sanitize_model_params_for_defaults(
        _extract_model_params(custom_model_info)
    )
    merged_model_params = deep_update(
        {**base_model_params},
        custom_model_params if isinstance(custom_model_params, dict) else {},
    )

    # Normalize request payload first so explicit nulls don't block default fallbacks.
    merged_payload = _normalize_payload_known_params_for_endpoint(merged_payload, endpoint)
    merged_model_params_for_defaults = (
        {**merged_model_params} if isinstance(merged_model_params, dict) else {}
    )
    merged_model_system = merged_model_params_for_defaults.pop("system", None)
    merged_model_defaults = apply_model_params_to_body_openai(
        merged_model_params_for_defaults, {}
    )

    # Treat empty request values as unset only for keys that have model defaults.
    default_keys: set[str] = set(merged_model_defaults.keys())
    for key in list(default_keys):
        if key in merged_payload and _is_empty_default_param_value(merged_payload[key]):
            merged_payload.pop(key, None)

    if endpoint == "responses":
        # /responses expects `input` + optional `instructions`.
        # Avoid injecting chat-only `messages` in upstream payload.
        for key, value in merged_model_defaults.items():
            if key not in merged_payload:
                merged_payload[key] = value

        if (
            isinstance(merged_model_system, str)
            and merged_model_system.strip()
            and _is_empty_default_param_value(merged_payload.get("instructions"))
        ):
            merged_payload["instructions"] = merged_model_system

        merged_payload.pop("messages", None)
    else:
        merged_payload = apply_model_params_as_defaults_openai(
            merged_model_params, merged_payload, metadata, user
        )

    merged_payload.pop("endpoint_kind", None)
    merged_payload.pop("endpointKind", None)
    merged_payload = _normalize_payload_known_params_for_endpoint(
        merged_payload, endpoint
    )
    return merged_payload


def get_provider_type(api_config: Optional[dict]) -> str:
    if not isinstance(api_config, dict):
        return "openai"

    provider_type = api_config.get("provider_type")
    if provider_type in {"openai", "openai_responses"}:
        return provider_type

    if provider_type == "azure_openai" or api_config.get("azure", False):
        return "openai"

    return "openai"


def normalize_openai_api_config(api_config: Optional[dict]) -> dict:
    if not isinstance(api_config, dict):
        return {}

    normalized = {**api_config}
    normalized["provider_type"] = get_provider_type(api_config)

    auth_type = normalized.get("auth_type")
    if auth_type in {"azure_ad", "microsoft_entra_id"}:
        normalized["auth_type"] = "bearer"

    # Azure OpenAI provider is removed. Drop legacy keys.
    normalized.pop("azure", None)
    normalized.pop("api_version", None)

    return normalized


def normalize_connection_type(connection_type: Optional[str]) -> str:
    # `local` connection types are deprecated and should behave as external.
    if connection_type == "local":
        return "external"
    return connection_type or "external"


def is_responses_provider(api_config: Optional[dict]) -> bool:
    return get_provider_type(api_config) == "openai_responses"


def _resolve_prompt_cache_key_for_completion_request(
    payload: dict, metadata: Optional[dict], user: UserModel
) -> Optional[str]:
    return adapter_resolve_prompt_cache_key(payload, metadata, user)


def _inject_prompt_cache_params_for_completion_request(
    provider_type: str,
    endpoint: str,
    payload: dict,
    metadata: Optional[dict],
    user: UserModel,
) -> None:
    apply_prompt_cache_policy(
        provider_type=provider_type,
        endpoint_kind=endpoint,
        payload=payload,
        metadata=metadata,
        user=user,
    )


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

    if token:
        headers["Authorization"] = f"Bearer {token}"

    if config.get("headers") and isinstance(config.get("headers"), dict):
        headers = {**headers, **config.get("headers")}

    return headers, cookies


##########################################
#
# API routes
#
##########################################

router = APIRouter()


@router.get("/config")
async def get_config(request: Request, user=Depends(get_admin_user)):
    normalized_api_configs = {
        key: normalize_openai_api_config(value)
        if isinstance(value, dict)
        else value
        for key, value in (request.app.state.config.OPENAI_API_CONFIGS or {}).items()
    }
    return {
        "ENABLE_OPENAI_API": request.app.state.config.ENABLE_OPENAI_API,
        "OPENAI_API_BASE_URLS": request.app.state.config.OPENAI_API_BASE_URLS,
        "OPENAI_API_KEYS": request.app.state.config.OPENAI_API_KEYS,
        "OPENAI_API_CONFIGS": normalized_api_configs,
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

    raw_api_configs = form_data.OPENAI_API_CONFIGS or {}
    normalized_api_configs = {}
    for key, value in raw_api_configs.items():
        if isinstance(value, dict):
            normalized_api_configs[key] = normalize_openai_api_config(value)
        else:
            normalized_api_configs[key] = value

    request.app.state.config.OPENAI_API_CONFIGS = normalized_api_configs

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
        api_config = normalize_openai_api_config(api_config)

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
            api_config = normalize_openai_api_config(api_config)
            api_config = normalize_openai_api_config(api_config)

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

            connection_type = normalize_connection_type(
                api_config.get("connection_type", "external")
            )
            prefix_id = api_config.get("prefix_id", None)
            tags = api_config.get("tags", [])
            provider_type = get_provider_type(api_config)

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
                model["provider_type"] = provider_type

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
                            "connection_type": normalize_connection_type(
                                model.get("connection_type", "external")
                            ),
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
        api_config = normalize_openai_api_config(api_config)

        r = None
        async with aiohttp.ClientSession(
            trust_env=True,
            timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST),
        ) as session:
            try:
                headers, cookies = await get_headers_and_cookies(
                    request, url, key, api_config, user=user
                )

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
    api_config = normalize_openai_api_config(api_config)

    async with aiohttp.ClientSession(
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST),
    ) as session:
        try:
            headers, cookies = await get_headers_and_cookies(
                request, url, key, api_config, user=user
            )

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


def is_openai_reasoning_model(model: str) -> bool:
    return model.lower().startswith(("o1", "o3", "o4", "gpt-5"))


def _validate_provider_for_endpoint(
    provider_type: str, endpoint: str, model_name: str
) -> None:
    if endpoint == "chat_completions" and provider_type == "openai_responses":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_name}' is configured with provider_type="
                "openai_responses and must use /responses."
            ),
        )

    if endpoint == "responses" and provider_type != "openai_responses":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_name}' is configured with provider_type="
                f"{provider_type} and must use /chat/completions."
            ),
        )


async def _generate_completion_with_endpoint(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
    bypass_filter: Optional[bool] = False,
    endpoint: str = "chat_completions",
):
    if BYPASS_MODEL_ACCESS_CONTROL:
        bypass_filter = True

    idx = 0

    payload = {**form_data}
    metadata = payload.pop("metadata", None)
    payload.pop("endpoint_kind", None)
    payload.pop("endpointKind", None)

    model_id = form_data.get("model")
    model_info = Models.get_model_by_id(model_id)
    base_model_info = None

    # Check model info and override the payload
    if model_info:
        if model_info.base_model_id:
            base_model_id = (
                request.base_model_id
                if hasattr(request, "base_model_id")
                else model_info.base_model_id
            )  # Use request's base_model_id if available
            base_model_info = Models.get_model_by_id(base_model_id)
            payload["model"] = base_model_id
            model_id = base_model_id

        payload = _build_payload_with_layered_model_defaults(
            payload=payload,
            endpoint=endpoint,
            metadata=metadata,
            user=user,
            custom_model_info=model_info,
            base_model_info=base_model_info,
        )

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

    if not model_info:
        payload = _normalize_payload_known_params_for_endpoint(payload, endpoint)

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
    api_config = normalize_openai_api_config(api_config)

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
    _validate_provider_for_endpoint(provider_type, endpoint, payload.get("model", ""))

    if endpoint == "responses" and "input" not in payload:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid /responses payload: missing required field 'input'. "
                "It looks like a chat-style payload ('messages')."
            ),
        )

    if not is_responses_provider(api_config):
        payload.pop("summary", None)
        if isinstance(payload.get("reasoning"), dict):
            payload["reasoning"].pop("summary", None)
            if not payload["reasoning"]:
                payload.pop("reasoning", None)

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

    if endpoint == "responses":
        _inject_prompt_cache_params_for_completion_request(
            provider_type,
            endpoint,
            payload,
            metadata,
            user,
        )
        request_url = f"{url}/responses"
    else:
        _inject_prompt_cache_params_for_completion_request(
            provider_type,
            endpoint,
            payload,
            metadata,
            user,
        )
        request_url = f"{url}/chat/completions"

    tools_shape_summary = "<none>"
    if isinstance(payload.get("tools"), list) and payload["tools"]:
        first_tool = payload["tools"][0]
        if isinstance(first_tool, dict):
            tools_shape_summary = ",".join(sorted(first_tool.keys()))
        else:
            tools_shape_summary = type(first_tool).__name__
    log.debug(
        "Completion upstream payload endpoint=%s provider=%s tools_shape=%s",
        endpoint,
        provider_type,
        tools_shape_summary,
    )

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


@router.post("/chat/completions")
async def generate_chat_completion(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
    bypass_filter: Optional[bool] = False,
):
    return await _generate_completion_with_endpoint(
        request=request,
        form_data=form_data,
        user=user,
        bypass_filter=bypass_filter,
        endpoint="chat_completions",
    )


@router.post("/responses")
async def generate_responses(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
    bypass_filter: Optional[bool] = False,
):
    return await _generate_completion_with_endpoint(
        request=request,
        form_data=form_data,
        user=user,
        bypass_filter=bypass_filter,
        endpoint="responses",
    )


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
    api_config = normalize_openai_api_config(api_config)

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
    api_config = normalize_openai_api_config(api_config)

    r = None
    session = None
    streaming = False

    try:
        headers, cookies = await get_headers_and_cookies(
            request, url, key, api_config, user=user
        )

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
