from open_webui.utils.task import prompt_template, prompt_variables_template
from open_webui.utils.misc import (
    deep_update,
    add_or_update_system_message,
    replace_system_message_content,
)

from typing import Callable, Optional, Any
import json


# inplace function: form_data is modified
def apply_system_prompt_to_body(
    system: Optional[str],
    form_data: dict,
    metadata: Optional[dict] = None,
    user=None,
    replace: bool = False,
    append: bool = False,
    separator: str = "\n",
) -> dict:
    if not system:
        return form_data

    # Metadata (WebUI Usage)
    if metadata:
        variables = metadata.get("variables", {})
        if variables:
            system = prompt_variables_template(system, variables)

    # Legacy (API Usage)
    system = prompt_template(system, user)

    if replace:
        form_data["messages"] = replace_system_message_content(
            system, form_data.get("messages", [])
        )
    else:
        form_data["messages"] = add_or_update_system_message(
            system,
            form_data.get("messages", []),
            append=append,
            separator=separator,
        )

    return form_data


# inplace function: form_data is modified
def apply_model_params_to_body(
    params: dict, form_data: dict, mappings: dict[str, Callable]
) -> dict:
    if not params:
        return form_data

    for key, value in params.items():
        if value is not None:
            if key in mappings:
                cast_func = mappings[key]
                if isinstance(cast_func, Callable):
                    form_data[key] = cast_func(value)
            else:
                form_data[key] = value

    return form_data


def remove_open_webui_params(params: dict) -> dict:
    """
    Removes OpenWebUI specific parameters from the provided dictionary.

    Args:
        params (dict): The dictionary containing parameters.

    Returns:
        dict: The modified dictionary with OpenWebUI parameters removed.
    """
    open_webui_params = {
        "stream_response": bool,
        "stream_delta_chunk_size": int,
        "function_calling": str,
        "reasoning_tags": list,
        "system": str,
    }

    for key in list(params.keys()):
        if key in open_webui_params:
            del params[key]

    return params


# inplace function: form_data is modified
def apply_model_params_to_body_openai(params: dict, form_data: dict) -> dict:
    params = remove_open_webui_params(params)

    custom_params = params.pop("custom_params", {})
    if custom_params:
        # Attempt to parse custom_params if they are strings
        for key, value in custom_params.items():
            if isinstance(value, str):
                try:
                    # Attempt to parse the string as JSON
                    custom_params[key] = json.loads(value)
                except json.JSONDecodeError:
                    # If it fails, keep the original string
                    pass

        # If there are custom parameters, we need to apply them first
        params = deep_update(params, custom_params)

    mappings = {
        "temperature": float,
        "top_p": float,
        "max_tokens": int,
        "frequency_penalty": float,
        "presence_penalty": float,
        "reasoning_effort": str,
        "verbosity": str,
        "summary": str,
        "seed": lambda x: x,
        "stop": lambda x: [bytes(s, "utf-8").decode("unicode_escape") for s in x],
        "logit_bias": lambda x: x,
        "response_format": dict,
    }
    return apply_model_params_to_body(params, form_data, mappings)


# inplace function: form_data is modified
def apply_model_params_as_defaults_openai(
    model_params: dict,
    form_data: dict,
    metadata: Optional[dict] = None,
    user=None,
) -> dict:
    """
    Applies model params as fallbacks only.

    Request/body values always take precedence over model-level defaults, including
    explicit nulls in the request body.
    """
    if not model_params:
        return form_data

    params = model_params.copy()
    system = params.pop("system", None)

    # Reuse the existing coercion/custom_params logic to build normalized defaults.
    defaults = apply_model_params_to_body_openai(params, {})
    for key, value in defaults.items():
        if key not in form_data:
            form_data[key] = value

    # Model system prompt always participates and is prepended when a request
    # system prompt already exists.
    if system:
        form_data = apply_system_prompt_to_body(
            system,
            form_data,
            metadata,
            user,
            append=False,
            separator="\n\n",
        )

    return form_data


def merge_model_params_with_base(
    model_info: Any,
    get_model_by_id: Optional[Callable[[str], Any]] = None,
) -> dict:
    """
    Build effective model default params with inheritance:
      base_model.params < custom_model.params
    """
    if not model_info:
        return {}

    if get_model_by_id is None:
        from open_webui.models.models import Models

        get_model_by_id = Models.get_model_by_id

    def _as_dict(value: Any) -> dict:
        if not value:
            return {}
        if hasattr(value, "model_dump"):
            dumped = value.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        if isinstance(value, dict):
            return {**value}
        return {}

    def _normalize_custom_params(raw: dict) -> dict:
        normalized: dict = {}
        for key, value in (raw or {}).items():
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, (dict, list)):
                        normalized[key] = parsed
                        continue
                except json.JSONDecodeError:
                    pass
            normalized[key] = value
        return normalized

    merged_params: dict = {}
    base_params_dict: dict = {}

    base_model_id = getattr(model_info, "base_model_id", None)
    if base_model_id:
        base_model_info = get_model_by_id(base_model_id)
        base_params_dict = _as_dict(
            getattr(base_model_info, "params", None) if base_model_info else None
        )
        merged_params = {**base_params_dict}

    model_params = _as_dict(getattr(model_info, "params", None))
    if model_params:
        merged_params = deep_update(merged_params, model_params)

    base_custom_params = _normalize_custom_params(base_params_dict.get("custom_params", {}))
    custom_custom_params = _normalize_custom_params(model_params.get("custom_params", {}))
    merged_custom_params = deep_update(base_custom_params, custom_custom_params)
    if merged_custom_params:
        merged_params["custom_params"] = merged_custom_params

    return merged_params
