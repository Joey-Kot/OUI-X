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
async def test_get_all_models_responses_normalizes_local_connection_type(monkeypatch):
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
                    OPENAI_API_BASE_URLS=["https://example.com/v1"],
                    OPENAI_API_KEYS=["sk-test"],
                    OPENAI_API_CONFIGS={
                        "0": {
                            "provider_type": "openai",
                            "connection_type": "local",
                        }
                    },
                )
            )
        )
    )

    responses = await openai_router.get_all_models_responses(request, _user())

    assert responses[0]["data"][0]["connection_type"] == "external"


@pytest.mark.asyncio
async def test_get_all_models_normalizes_local_connection_type(monkeypatch):
    async def fake_get_all_models_responses(_request, user=None):
        return [
            {
                "object": "list",
                "data": [{"id": "gpt-5-mini", "connection_type": "local"}],
            }
        ]

    monkeypatch.setattr(openai_router, "get_all_models_responses", fake_get_all_models_responses)

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                config=SimpleNamespace(
                    ENABLE_OPENAI_API=True,
                    OPENAI_API_BASE_URLS=["https://example.com/v1"],
                )
            )
        )
    )

    models = await openai_router.get_all_models.__wrapped__(request, _user())

    assert models["data"][0]["connection_type"] == "external"


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


@pytest.mark.asyncio
async def test_generate_responses_applies_model_defaults_when_request_has_nulls(monkeypatch):
    captured = {}

    class FakeResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"Content-Type": "application/json"}

        async def json(self):
            return {"id": "resp_456", "object": "response", "ok": True}

        async def text(self):
            return ""

        def close(self):
            return None

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def request(self, method, url, data, headers, cookies, ssl):
            captured["payload"] = json.loads(data)
            return FakeResponse()

        async def close(self):
            return None

    async def fake_get_all_models(_request, user=None):
        return {"data": []}

    async def fake_headers_and_cookies(*args, **kwargs):
        return {}, {}

    model_info = SimpleNamespace(
        base_model_id=None,
        params=SimpleNamespace(model_dump=lambda: {"temperature": 0.2, "max_tokens": 256}),
        user_id="admin",
        access_control=None,
    )

    monkeypatch.setattr(openai_router, "get_all_models", fake_get_all_models)
    monkeypatch.setattr(openai_router, "get_headers_and_cookies", fake_headers_and_cookies)
    monkeypatch.setattr(openai_router.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(openai_router.Models, "get_model_by_id", lambda _model_id: model_info)

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
        "temperature": None,
        "max_tokens": None,
    }

    response = await openai_router.generate_responses(
        request=request,
        form_data=form_data,
        user=_user(role="admin"),
    )

    assert response["id"] == "resp_456"
    assert captured["payload"]["temperature"] == 0.2
    assert captured["payload"]["max_output_tokens"] == 256
    assert "max_tokens" not in captured["payload"]


@pytest.mark.asyncio
async def test_generate_responses_maps_legacy_known_fields_and_preserves_unknown_fields(
    monkeypatch,
):
    captured = {}

    class FakeResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"Content-Type": "application/json"}

        async def json(self):
            return {"id": "resp_789", "object": "response", "ok": True}

        async def text(self):
            return ""

        def close(self):
            return None

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def request(self, method, url, data, headers, cookies, ssl):
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
        "reasoning_effort": "medium",
        "summary": "auto",
        "verbosity": "high",
        "response_format": {"type": "json_object"},
        "max_tokens": 1024,
        "future_unknown_field": {"x": 1},
    }

    response = await openai_router.generate_responses(
        request=request,
        form_data=form_data,
        user=_user(role="admin"),
    )

    assert response["id"] == "resp_789"
    assert captured["payload"]["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert captured["payload"]["text"] == {
        "verbosity": "high",
        "format": {"type": "json_object"},
    }
    assert captured["payload"]["max_output_tokens"] == 1024
    assert "reasoning_effort" not in captured["payload"]
    assert "summary" not in captured["payload"]
    assert "verbosity" not in captured["payload"]
    assert "response_format" not in captured["payload"]
    assert "max_tokens" not in captured["payload"]
    assert captured["payload"]["future_unknown_field"] == {"x": 1}


@pytest.mark.asyncio
async def test_generate_chat_completions_preserves_reasoning_effort_and_verbosity(
    monkeypatch,
):
    captured = {}

    class FakeResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"Content-Type": "application/json"}

        async def json(self):
            return {"id": "chatcmpl_123", "object": "chat.completion", "ok": True}

        async def text(self):
            return ""

        def close(self):
            return None

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def request(self, method, url, data, headers, cookies, ssl):
            captured["payload"] = json.loads(data)
            captured["url"] = url
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
                OPENAI_MODELS={"gpt-5.4": {"urlIdx": 0}},
                config=SimpleNamespace(
                    OPENAI_API_BASE_URLS=["https://api.openai.com/v1"],
                    OPENAI_API_KEYS=["sk-test"],
                    OPENAI_API_CONFIGS={"0": {"provider_type": "openai"}},
                ),
            )
        )
    )

    form_data = {
        "model": "gpt-5.4",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "reasoning_effort": "low",
        "verbosity": "high",
        "future_unknown_field": {"debug": True},
    }

    response = await openai_router.generate_chat_completion(
        request=request,
        form_data=form_data,
        user=_user(role="admin"),
    )

    assert response["id"] == "chatcmpl_123"
    assert captured["url"].endswith("/chat/completions")
    assert captured["payload"]["reasoning_effort"] == "low"
    assert captured["payload"]["verbosity"] == "high"
    assert captured["payload"]["future_unknown_field"] == {"debug": True}


@pytest.mark.asyncio
async def test_generate_chat_completions_known_nulls_allow_model_default_fallback(
    monkeypatch,
):
    captured = {}

    class FakeResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"Content-Type": "application/json"}

        async def json(self):
            return {"id": "chatcmpl_456", "object": "chat.completion", "ok": True}

        async def text(self):
            return ""

        def close(self):
            return None

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def request(self, method, url, data, headers, cookies, ssl):
            captured["payload"] = json.loads(data)
            return FakeResponse()

        async def close(self):
            return None

    async def fake_get_all_models(_request, user=None):
        return {"data": []}

    async def fake_headers_and_cookies(*args, **kwargs):
        return {}, {}

    model_info = SimpleNamespace(
        base_model_id=None,
        params=SimpleNamespace(
            model_dump=lambda: {
                "reasoning_effort": "low",
                "verbosity": "high",
                "temperature": 0.2,
            }
        ),
        user_id="admin",
        access_control=None,
    )

    monkeypatch.setattr(openai_router, "get_all_models", fake_get_all_models)
    monkeypatch.setattr(openai_router, "get_headers_and_cookies", fake_headers_and_cookies)
    monkeypatch.setattr(openai_router.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(openai_router.Models, "get_model_by_id", lambda _model_id: model_info)

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                OPENAI_MODELS={"gpt-5.4": {"urlIdx": 0}},
                config=SimpleNamespace(
                    OPENAI_API_BASE_URLS=["https://api.openai.com/v1"],
                    OPENAI_API_KEYS=["sk-test"],
                    OPENAI_API_CONFIGS={"0": {"provider_type": "openai"}},
                ),
            )
        )
    )

    form_data = {
        "model": "gpt-5.4",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "reasoning_effort": None,
        "verbosity": None,
        "temperature": None,
        "future_unknown_field": None,
    }

    response = await openai_router.generate_chat_completion(
        request=request,
        form_data=form_data,
        user=_user(role="admin"),
    )

    assert response["id"] == "chatcmpl_456"
    assert captured["payload"]["reasoning_effort"] == "low"
    assert captured["payload"]["verbosity"] == "high"
    assert captured["payload"]["temperature"] == 0.2
    # Unknown keys remain passthrough, including nulls.
    assert "future_unknown_field" in captured["payload"]
    assert captured["payload"]["future_unknown_field"] is None


@pytest.mark.asyncio
async def test_endpoint_kind_fields_are_not_forwarded_upstream(monkeypatch):
    captured = {}

    class FakeResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"Content-Type": "application/json"}

        async def json(self):
            return {"id": "ok_1", "object": "response", "ok": True}

        async def text(self):
            return ""

        def close(self):
            return None

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def request(self, method, url, data, headers, cookies, ssl):
            captured["payload"] = json.loads(data)
            captured["url"] = url
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

    # responses path
    request_responses = SimpleNamespace(
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

    await openai_router.generate_responses(
        request=request_responses,
        form_data={
            "model": "gpt-5-mini",
            "input": [{"role": "user", "content": "hello"}],
            "stream": False,
            "endpoint_kind": "responses",
            "endpointKind": "responses",
        },
        user=_user(role="admin"),
    )
    assert captured["url"].endswith("/responses")
    assert "endpoint_kind" not in captured["payload"]
    assert "endpointKind" not in captured["payload"]

    # chat completions path
    request_chat = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                OPENAI_MODELS={"gpt-5.4": {"urlIdx": 0}},
                config=SimpleNamespace(
                    OPENAI_API_BASE_URLS=["https://api.openai.com/v1"],
                    OPENAI_API_KEYS=["sk-test"],
                    OPENAI_API_CONFIGS={"0": {"provider_type": "openai"}},
                ),
            )
        )
    )

    await openai_router.generate_chat_completion(
        request=request_chat,
        form_data={
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "endpoint_kind": "chat_completions",
            "endpointKind": "chat_completions",
        },
        user=_user(role="admin"),
    )
    assert captured["url"].endswith("/chat/completions")
    assert "endpoint_kind" not in captured["payload"]
    assert "endpointKind" not in captured["payload"]


@pytest.mark.asyncio
async def test_generate_chat_completions_custom_model_uses_layered_defaults(monkeypatch):
    captured = {}

    class FakeResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"Content-Type": "application/json"}

        async def json(self):
            return {"id": "chatcmpl_layered_1", "object": "chat.completion", "ok": True}

        async def text(self):
            return ""

        def close(self):
            return None

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def request(self, method, url, data, headers, cookies, ssl):
            captured["payload"] = json.loads(data)
            captured["url"] = url
            return FakeResponse()

        async def close(self):
            return None

    async def fake_get_all_models(_request, user=None):
        return {"data": []}

    async def fake_headers_and_cookies(*args, **kwargs):
        return {}, {}

    def fake_get_model_by_id(model_id: str):
        if model_id == "custom-model":
            return SimpleNamespace(
                base_model_id="gpt-5.4",
                params=SimpleNamespace(
                    model_dump=lambda: {
                        "temperature": 0.7,
                        "top_p": 0.6,
                        "presence_penalty": "",
                    }
                ),
                user_id="admin",
                access_control=None,
            )
        if model_id == "gpt-5.4":
            return SimpleNamespace(
                base_model_id=None,
                params=SimpleNamespace(
                    model_dump=lambda: {
                        "temperature": 0.2,
                        "top_p": 0.3,
                        "presence_penalty": 1.0,
                        "frequency_penalty": 0.8,
                    }
                ),
                user_id="admin",
                access_control=None,
            )
        return None

    monkeypatch.setattr(openai_router, "get_all_models", fake_get_all_models)
    monkeypatch.setattr(openai_router, "get_headers_and_cookies", fake_headers_and_cookies)
    monkeypatch.setattr(openai_router.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(openai_router.Models, "get_model_by_id", fake_get_model_by_id)

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                OPENAI_MODELS={"gpt-5.4": {"urlIdx": 0}},
                config=SimpleNamespace(
                    OPENAI_API_BASE_URLS=["https://api.openai.com/v1"],
                    OPENAI_API_KEYS=["sk-test"],
                    OPENAI_API_CONFIGS={"0": {"provider_type": "openai"}},
                ),
            )
        )
    )

    form_data = {
        "model": "custom-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "temperature": 0.9,
        "top_p": "",
        "presence_penalty": "",
        "frequency_penalty": 0.0,
    }

    response = await openai_router.generate_chat_completion(
        request=request,
        form_data=form_data,
        user=_user(role="admin"),
    )

    assert response["id"] == "chatcmpl_layered_1"
    assert captured["url"].endswith("/chat/completions")
    assert captured["payload"]["model"] == "gpt-5.4"
    assert captured["payload"]["temperature"] == 0.9
    assert captured["payload"]["top_p"] == 0.6
    assert captured["payload"]["presence_penalty"] == 1.0
    assert captured["payload"]["frequency_penalty"] == 0.0


@pytest.mark.asyncio
async def test_generate_chat_completions_empty_container_values_do_not_block_fallback(
    monkeypatch,
):
    captured = {}

    class FakeResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"Content-Type": "application/json"}

        async def json(self):
            return {"id": "chatcmpl_layered_2", "object": "chat.completion", "ok": True}

        async def text(self):
            return ""

        def close(self):
            return None

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def request(self, method, url, data, headers, cookies, ssl):
            captured["payload"] = json.loads(data)
            return FakeResponse()

        async def close(self):
            return None

    async def fake_get_all_models(_request, user=None):
        return {"data": []}

    async def fake_headers_and_cookies(*args, **kwargs):
        return {}, {}

    def fake_get_model_by_id(model_id: str):
        if model_id == "custom-model":
            return SimpleNamespace(
                base_model_id="gpt-5.4",
                params=SimpleNamespace(
                    model_dump=lambda: {
                        "stop": [],
                        "logit_bias": {},
                    }
                ),
                user_id="admin",
                access_control=None,
            )
        if model_id == "gpt-5.4":
            return SimpleNamespace(
                base_model_id=None,
                params=SimpleNamespace(
                    model_dump=lambda: {
                        "stop": ["END"],
                        "logit_bias": {"1": 2},
                    }
                ),
                user_id="admin",
                access_control=None,
            )
        return None

    monkeypatch.setattr(openai_router, "get_all_models", fake_get_all_models)
    monkeypatch.setattr(openai_router, "get_headers_and_cookies", fake_headers_and_cookies)
    monkeypatch.setattr(openai_router.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(openai_router.Models, "get_model_by_id", fake_get_model_by_id)

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                OPENAI_MODELS={"gpt-5.4": {"urlIdx": 0}},
                config=SimpleNamespace(
                    OPENAI_API_BASE_URLS=["https://api.openai.com/v1"],
                    OPENAI_API_KEYS=["sk-test"],
                    OPENAI_API_CONFIGS={"0": {"provider_type": "openai"}},
                ),
            )
        )
    )

    form_data = {
        "model": "custom-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "stop": [],
    }

    response = await openai_router.generate_chat_completion(
        request=request,
        form_data=form_data,
        user=_user(role="admin"),
    )

    assert response["id"] == "chatcmpl_layered_2"
    assert captured["payload"]["stop"] == ["END"]
    assert captured["payload"]["logit_bias"] == {"1": 2}


@pytest.mark.asyncio
async def test_generate_responses_model_system_maps_to_instructions_and_no_messages(
    monkeypatch,
):
    captured = {}

    class FakeResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"Content-Type": "application/json"}

        async def json(self):
            return {"id": "resp_sys_1", "object": "response", "ok": True}

        async def text(self):
            return ""

        def close(self):
            return None

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def request(self, method, url, data, headers, cookies, ssl):
            captured["payload"] = json.loads(data)
            captured["url"] = url
            return FakeResponse()

        async def close(self):
            return None

    async def fake_get_all_models(_request, user=None):
        return {"data": []}

    async def fake_headers_and_cookies(*args, **kwargs):
        return {}, {}

    def fake_get_model_by_id(model_id: str):
        if model_id == "custom-model":
            return SimpleNamespace(
                base_model_id="gpt-5-mini",
                params=SimpleNamespace(
                    model_dump=lambda: {
                        "system": "you are custom system",
                    }
                ),
                user_id="admin",
                access_control=None,
            )
        return None

    monkeypatch.setattr(openai_router, "get_all_models", fake_get_all_models)
    monkeypatch.setattr(openai_router, "get_headers_and_cookies", fake_headers_and_cookies)
    monkeypatch.setattr(openai_router.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(openai_router.Models, "get_model_by_id", fake_get_model_by_id)

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
        "model": "custom-model",
        "input": [{"role": "user", "content": "who are you?"}],
        "stream": False,
    }

    response = await openai_router.generate_responses(
        request=request,
        form_data=form_data,
        user=_user(role="admin"),
    )

    assert response["id"] == "resp_sys_1"
    assert captured["url"].endswith("/responses")
    assert captured["payload"]["instructions"] == "you are custom system"
    assert "messages" not in captured["payload"]


@pytest.mark.asyncio
async def test_generate_responses_request_instructions_override_model_system(monkeypatch):
    captured = {}

    class FakeResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"Content-Type": "application/json"}

        async def json(self):
            return {"id": "resp_sys_2", "object": "response", "ok": True}

        async def text(self):
            return ""

        def close(self):
            return None

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def request(self, method, url, data, headers, cookies, ssl):
            captured["payload"] = json.loads(data)
            return FakeResponse()

        async def close(self):
            return None

    async def fake_get_all_models(_request, user=None):
        return {"data": []}

    async def fake_headers_and_cookies(*args, **kwargs):
        return {}, {}

    def fake_get_model_by_id(model_id: str):
        if model_id == "custom-model":
            return SimpleNamespace(
                base_model_id="gpt-5-mini",
                params=SimpleNamespace(model_dump=lambda: {"system": "model-system"}),
                user_id="admin",
                access_control=None,
            )
        return None

    monkeypatch.setattr(openai_router, "get_all_models", fake_get_all_models)
    monkeypatch.setattr(openai_router, "get_headers_and_cookies", fake_headers_and_cookies)
    monkeypatch.setattr(openai_router.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(openai_router.Models, "get_model_by_id", fake_get_model_by_id)

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

    response = await openai_router.generate_responses(
        request=request,
        form_data={
            "model": "custom-model",
            "input": [{"role": "user", "content": "who are you?"}],
            "instructions": "request-instructions",
            "stream": False,
        },
        user=_user(role="admin"),
    )

    assert response["id"] == "resp_sys_2"
    assert captured["payload"]["instructions"] == "request-instructions"
