import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from open_webui.routers import openai as openai_router


def _user(user_id: str = "user-1", role: str = "user"):
    return SimpleNamespace(
        id=user_id,
        role=role,
        name="Test User",
        email="test@example.com",
    )


def test_validate_provider_for_endpoint_rejects_responses_on_chat_endpoint():
    with pytest.raises(HTTPException) as exc:
        openai_router._validate_provider_for_endpoint(
            provider_type="openai_responses",
            endpoint="chat_completions",
            model_name="gpt-5",
        )

    assert exc.value.status_code == 400
    assert "/responses" in str(exc.value.detail)


def test_validate_provider_for_endpoint_rejects_chat_provider_on_responses_endpoint():
    with pytest.raises(HTTPException) as exc:
        openai_router._validate_provider_for_endpoint(
            provider_type="openai",
            endpoint="responses",
            model_name="gpt-4.1",
        )

    assert exc.value.status_code == 400
    assert "/chat/completions" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_get_all_models_responses_includes_provider_type(monkeypatch):
    async def fake_send_get_request(_url, _key=None, user=None):
        return {
            "object": "list",
            "data": [{"id": "gpt-5-mini", "owned_by": "openai"}],
        }

    monkeypatch.setattr(openai_router, "send_get_request", fake_send_get_request)

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                config=SimpleNamespace(
                    ENABLE_OPENAI_API=True,
                    OPENAI_API_BASE_URLS=["https://api.openai.com/v1"],
                    OPENAI_API_KEYS=["sk-test"],
                    OPENAI_API_CONFIGS={
                        "0": {
                            "provider_type": "openai_responses",
                            "connection_type": "external",
                        }
                    },
                )
            )
        )
    )

    responses = await openai_router.get_all_models_responses(request, _user())

    assert responses[0]["data"][0]["provider_type"] == "openai_responses"


@pytest.mark.asyncio
async def test_generate_responses_passthrough_preserves_unknown_fields(monkeypatch):
    captured = {}

    class FakeResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"Content-Type": "application/json"}

        async def json(self):
            return {"id": "resp_123", "object": "response", "ok": True}

        async def text(self):
            return ""

        def close(self):
            return None

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def request(self, method, url, data, headers, cookies, ssl):
            captured["method"] = method
            captured["url"] = url
            captured["payload"] = json.loads(data)
            return FakeResponse()

        async def close(self):
            return None

    async def fake_get_all_models(_request, user=None):
        return {"data": []}

    async def fake_headers_and_cookies(*args, **kwargs):
        return {}, {}

    monkeypatch.setattr(openai_router, "get_all_models", fake_get_all_models)
    monkeypatch.setattr(openai_router, "get_headers_and_cookies", fake_headers_and_cookies)
    monkeypatch.setattr(openai_router.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(openai_router.Models, "get_model_by_id", lambda _model_id: None)

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                OPENAI_MODELS={"gpt-5-mini": {"urlIdx": 0}},
                config=SimpleNamespace(
                    OPENAI_API_BASE_URLS=["https://api.openai.com/v1"],
                    OPENAI_API_KEYS=["sk-test"],
                    OPENAI_API_CONFIGS={"0": {"provider_type": "openai_responses"}},
                ),
            )
        )
    )

    form_data = {
        "model": "gpt-5-mini",
        "input": [{"role": "user", "content": "hello"}],
        "stream": False,
        "metadata": {"nested": {"safe": True}},
        "future_unknown_field": {"x": 1, "y": [1, 2, 3]},
    }

    response = await openai_router.generate_responses(
        request=request,
        form_data=form_data,
        user=_user(role="admin"),
    )

    assert response["id"] == "resp_123"
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/responses")
    assert captured["payload"]["future_unknown_field"] == {"x": 1, "y": [1, 2, 3]}
