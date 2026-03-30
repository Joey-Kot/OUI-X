from types import SimpleNamespace

import pytest

from open_webui.routers import tasks as tasks_router


def _build_request_with_config():
    config = SimpleNamespace(
        TASK_MODEL="",
        TITLE_GENERATION_PROMPT_TEMPLATE="",
        IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE="",
        ENABLE_AUTOCOMPLETE_GENERATION=True,
        AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH=-1,
        TAGS_GENERATION_PROMPT_TEMPLATE="",
        FOLLOW_UP_GENERATION_PROMPT_TEMPLATE="",
        ENABLE_FOLLOW_UP_GENERATION=True,
        ENABLE_TAGS_GENERATION=True,
        ENABLE_TITLE_GENERATION=True,
        ENABLE_SEARCH_QUERY_GENERATION=True,
        ENABLE_RETRIEVAL_QUERY_GENERATION=True,
        RETRIEVAL_QUERY_GENERATION_REFER_CONTEXT_TURNS=3,
        QUERY_GENERATION_PROMPT_TEMPLATE="",
        TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE="",
        VOICE_MODE_PROMPT_TEMPLATE="",
    )
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(config=config)))


def _base_task_config_payload():
    return {
        "TASK_MODEL": "task-model-id",
        "ENABLE_TITLE_GENERATION": True,
        "TITLE_GENERATION_PROMPT_TEMPLATE": "title",
        "IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE": "image",
        "ENABLE_AUTOCOMPLETE_GENERATION": True,
        "AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH": 128,
        "TAGS_GENERATION_PROMPT_TEMPLATE": "tags",
        "FOLLOW_UP_GENERATION_PROMPT_TEMPLATE": "follow-up",
        "ENABLE_FOLLOW_UP_GENERATION": True,
        "ENABLE_TAGS_GENERATION": True,
        "ENABLE_SEARCH_QUERY_GENERATION": True,
        "ENABLE_RETRIEVAL_QUERY_GENERATION": True,
        "RETRIEVAL_QUERY_GENERATION_REFER_CONTEXT_TURNS": 3,
        "QUERY_GENERATION_PROMPT_TEMPLATE": "query",
        "TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE": "tools",
        "VOICE_MODE_PROMPT_TEMPLATE": "voice",
    }


@pytest.mark.asyncio
async def test_get_task_config_does_not_expose_legacy_task_model_external():
    request = _build_request_with_config()

    response = await tasks_router.get_task_config(request, user=SimpleNamespace())

    assert "TASK_MODEL" in response
    assert "TASK_MODEL_EXTERNAL" not in response


@pytest.mark.asyncio
async def test_update_task_config_updates_task_model():
    request = _build_request_with_config()
    payload = _base_task_config_payload()
    form_data = tasks_router.TaskConfigForm.model_validate(payload)

    response = await tasks_router.update_task_config(
        request, form_data=form_data, user=SimpleNamespace()
    )

    assert request.app.state.config.TASK_MODEL == "task-model-id"
    assert response["TASK_MODEL"] == "task-model-id"
    assert "TASK_MODEL_EXTERNAL" not in response


@pytest.mark.asyncio
async def test_update_task_config_maps_legacy_task_model_external_to_task_model():
    request = _build_request_with_config()
    payload = _base_task_config_payload()
    payload["TASK_MODEL"] = ""
    payload["TASK_MODEL_EXTERNAL"] = "legacy-external-task-model-id"
    form_data = tasks_router.TaskConfigForm.model_validate(payload)

    response = await tasks_router.update_task_config(
        request, form_data=form_data, user=SimpleNamespace()
    )

    assert request.app.state.config.TASK_MODEL == "legacy-external-task-model-id"
    assert response["TASK_MODEL"] == "legacy-external-task-model-id"
