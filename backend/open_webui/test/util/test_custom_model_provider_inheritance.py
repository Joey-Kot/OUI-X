from types import SimpleNamespace

import pytest

from open_webui.utils import models as models_utils


class _Dumpable:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


class _CustomModel:
    def __init__(
        self,
        *,
        model_id: str,
        base_model_id: str,
        name: str = "Custom",
        created_at: int = 123,
        user_id: str = "user-1",
        is_active: bool = True,
        meta: dict | None = None,
    ):
        self.id = model_id
        self.user_id = user_id
        self.base_model_id = base_model_id
        self.name = name
        self.is_active = is_active
        self.created_at = created_at
        self.meta = _Dumpable(meta or {})

    def model_dump(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "base_model_id": self.base_model_id,
            "name": self.name,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "meta": self.meta.model_dump(),
            "params": {"temperature": 0.3},
        }


def _request_state():
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                MODELS={},
                BASE_MODELS={},
                config=SimpleNamespace(ENABLE_BASE_MODELS_CACHE=False),
            )
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_provider_type,expected_provider_type",
    [
        ("openai_responses", "openai_responses"),
        ("openai", "openai"),
        (None, "openai"),
    ],
)
async def test_custom_model_inherits_provider_type_from_base_model(
    monkeypatch, base_provider_type, expected_provider_type
):
    base_model = {
        "id": "base-model",
        "name": "Base Model",
        "owned_by": "openai",
        "connection_type": "external",
        "openai": {"id": "base-model", "owned_by": "openai"},
        "urlIdx": 2,
        **({"provider_type": base_provider_type} if base_provider_type else {}),
    }

    async def _fake_get_all_base_models(_request, user=None):
        return [base_model]

    monkeypatch.setattr(models_utils, "get_all_base_models", _fake_get_all_base_models)
    monkeypatch.setattr(
        models_utils.Models,
        "get_all_models",
        lambda: [_CustomModel(model_id="custom-model", base_model_id="base-model")],
    )

    monkeypatch.setattr(models_utils.Functions, "get_global_action_functions", lambda: [])
    monkeypatch.setattr(models_utils.Functions, "get_global_filter_functions", lambda: [])
    monkeypatch.setattr(
        models_utils.Functions,
        "get_functions_by_type",
        lambda *_args, **_kwargs: [],
    )

    request = _request_state()
    user = SimpleNamespace(id="user-1")

    result = await models_utils.get_all_models(request=request, refresh=True, user=user)

    custom = next((m for m in result if m.get("id") == "custom-model"), None)

    assert custom is not None
    assert custom["provider_type"] == expected_provider_type
    assert custom["info"]["provider_type"] == expected_provider_type
    assert custom["openai"] == {"id": "base-model", "owned_by": "openai"}
    assert custom["urlIdx"] == 2
