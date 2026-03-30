from types import SimpleNamespace

import pytest

from open_webui.routers import tasks as tasks_router


@pytest.mark.asyncio
async def test_generate_task_completion_wraps_responses_payload(monkeypatch):
    response_payload = {
        "id": "resp_test",
        "object": "response",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "wrapped content",
                    }
                ],
            }
        ],
    }

    async def fake_generate_responses(_request, form_data, user):
        assert form_data.get("endpoint_kind") == "responses"
        return response_payload

    monkeypatch.setattr(tasks_router, "generate_responses", fake_generate_responses)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(OPENAI_MODELS={})))
    models = {
        "test-model": {
            "id": "test-model",
            "provider_type": "openai_responses",
        }
    }
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    user = SimpleNamespace(role="admin")

    result = await tasks_router._generate_task_completion(request, payload, user, models)

    assert result["choices"][0]["message"]["content"] == "wrapped content"
    assert result["raw_response"]["id"] == "resp_test"

