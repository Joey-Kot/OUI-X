from types import SimpleNamespace

from fastapi.testclient import TestClient

from test.util.mock_user import mock_user


def test_api_responses_endpoint_routes_to_responses_handler(monkeypatch):
    from open_webui import main as main_module

    app = main_module.app

    called = {}

    async def fake_get_all_models(_request, user=None):
        return {"data": []}

    async def fake_responses_handler(_request, form_data, _user):
        called["form_data"] = dict(form_data)
        return {"ok": True, "endpoint": "responses", "payload": form_data}

    monkeypatch.setattr(main_module, "get_all_models", fake_get_all_models)
    monkeypatch.setattr(main_module, "responses_handler", fake_responses_handler)

    body = {
        "model": "gpt-5-mini",
        "input": [{"role": "user", "content": "hi"}],
        "stream": False,
        "model_item": {
            "id": "gpt-5-mini",
            "direct": True,
            "provider_type": "openai_responses",
        },
    }

    with mock_user(app, id="u1", role="user"):
        with TestClient(app) as client:
            response = client.post("/api/responses", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["endpoint"] == "responses"
    assert called["form_data"]["model"] == "gpt-5-mini"
    assert "model_item" not in called["form_data"]


def test_api_chat_completions_routes_to_chat_handler_pipeline(monkeypatch):
    from open_webui import main as main_module

    app = main_module.app

    async def fake_get_all_models(_request, user=None):
        return {"data": []}

    async def fake_process_chat_payload(_request, form_data, _user, metadata, _model):
        return form_data, metadata, []

    async def fake_chat_handler(_request, form_data, _user):
        return {"upstream": "ok", "echo_model": form_data.get("model")}

    async def fake_process_chat_response(
        _request,
        response,
        form_data,
        _user,
        _metadata,
        _model,
        _events,
        _tasks,
    ):
        return {
            "ok": True,
            "endpoint": "chat_completions",
            "upstream": response,
            "form_data": form_data,
        }

    monkeypatch.setattr(main_module, "get_all_models", fake_get_all_models)
    monkeypatch.setattr(main_module, "process_chat_payload", fake_process_chat_payload)
    monkeypatch.setattr(main_module, "chat_completion_handler", fake_chat_handler)
    monkeypatch.setattr(main_module, "process_chat_response", fake_process_chat_response)

    body = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "model_item": {
            "id": "gpt-4.1-mini",
            "direct": True,
            "provider_type": "openai",
            "info": {"meta": {"capabilities": {"usage": True}}},
        },
    }

    with mock_user(app, id="u2", role="user"):
        with TestClient(app) as client:
            response = client.post("/api/chat/completions", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["endpoint"] == "chat_completions"
    assert payload["upstream"]["upstream"] == "ok"
    assert payload["upstream"]["echo_model"] == "gpt-4.1-mini"
