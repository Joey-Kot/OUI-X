from types import SimpleNamespace

from open_webui.routers import openai as openai_router


def _user(user_id: str = "user-1", role: str = "user"):
    return SimpleNamespace(id=user_id, role=role)


def test_resolver_uses_explicit_prompt_cache_key_and_skips_chat_lookup(monkeypatch):
    class DummyChats:
        def get_chat_by_id_and_user_id(self, *_args, **_kwargs):
            raise AssertionError("should not query chats when explicit key is present")

        def get_chat_by_id(self, *_args, **_kwargs):
            raise AssertionError("should not query chats when explicit key is present")

        def update_chat_metadata_by_id(self, *_args, **_kwargs):
            raise AssertionError("should not update chats when explicit key is present")

    monkeypatch.setattr(openai_router, "Chats", DummyChats())

    resolved = openai_router._resolve_prompt_cache_key_for_responses(
        {"prompt_cache_key": "request-key"},
        {"chat_id": "chat-1"},
        _user(),
    )

    assert resolved == "request-key"


def test_resolver_reuses_persisted_chat_meta_key(monkeypatch):
    class DummyChats:
        def get_chat_by_id_and_user_id(self, *_args, **_kwargs):
            return SimpleNamespace(meta={"prompt_cache_key": "pc:v1:stored"})

        def get_chat_by_id(self, *_args, **_kwargs):
            return None

        def update_chat_metadata_by_id(self, *_args, **_kwargs):
            raise AssertionError("should not update when key already exists")

    monkeypatch.setattr(openai_router, "Chats", DummyChats())

    resolved = openai_router._resolve_prompt_cache_key_for_responses(
        {},
        {"chat_id": "chat-1"},
        _user(),
    )

    assert resolved == "pc:v1:stored"


def test_resolver_generates_and_persists_for_chat_without_key(monkeypatch):
    calls = []

    class DummyChats:
        def get_chat_by_id_and_user_id(self, chat_id, user_id):
            calls.append(("get_chat_by_id_and_user_id", chat_id, user_id))
            return SimpleNamespace(meta={})

        def get_chat_by_id(self, *_args, **_kwargs):
            calls.append(("get_chat_by_id",))
            return None

        def update_chat_metadata_by_id(self, chat_id, meta):
            calls.append(("update_chat_metadata_by_id", chat_id, meta))
            return SimpleNamespace(meta=meta)

    monkeypatch.setattr(openai_router, "Chats", DummyChats())

    resolved = openai_router._resolve_prompt_cache_key_for_responses(
        {},
        {"chat_id": "chat-1"},
        _user(),
    )

    assert resolved.startswith("pc:v1:")
    assert len(resolved) > len("pc:v1:")
    update_call = calls[-1]
    assert update_call[0] == "update_chat_metadata_by_id"
    assert update_call[1] == "chat-1"
    assert update_call[2]["prompt_cache_key"] == resolved
    assert update_call[2]["prompt_cache_key_version"] == "v1"


def test_resolver_returns_generated_key_when_persist_fails(monkeypatch):
    class DummyChats:
        def get_chat_by_id_and_user_id(self, *_args, **_kwargs):
            return SimpleNamespace(meta={})

        def get_chat_by_id(self, *_args, **_kwargs):
            return None

        def update_chat_metadata_by_id(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(openai_router, "Chats", DummyChats())

    resolved = openai_router._resolve_prompt_cache_key_for_responses(
        {},
        {"chat_id": "chat-1"},
        _user(),
    )

    assert resolved.startswith("pc:v1:")


def test_resolver_derives_deterministic_key_for_local_chat():
    first = openai_router._resolve_prompt_cache_key_for_responses(
        {},
        {"chat_id": "local:session-a"},
        _user(),
    )
    second = openai_router._resolve_prompt_cache_key_for_responses(
        {},
        {"chat_id": "local:session-a"},
        _user(),
    )
    third = openai_router._resolve_prompt_cache_key_for_responses(
        {},
        {"chat_id": "local:session-b"},
        _user(),
    )

    assert first == second
    assert first != third
    assert first.startswith("pc:v1:local:")


def test_resolver_returns_none_without_chat_id():
    resolved = openai_router._resolve_prompt_cache_key_for_responses(
        {},
        {},
        _user(),
    )

    assert resolved is None


def test_resolver_uses_admin_fallback_chat_lookup(monkeypatch):
    calls = []

    class DummyChats:
        def get_chat_by_id_and_user_id(self, *_args, **_kwargs):
            calls.append("by_user")
            return None

        def get_chat_by_id(self, *_args, **_kwargs):
            calls.append("by_id")
            return SimpleNamespace(meta={"prompt_cache_key": "pc:v1:admin-stored"})

        def update_chat_metadata_by_id(self, *_args, **_kwargs):
            raise AssertionError("should not update when fallback chat has key")

    monkeypatch.setattr(openai_router, "Chats", DummyChats())

    resolved = openai_router._resolve_prompt_cache_key_for_responses(
        {},
        {"chat_id": "chat-2"},
        _user(user_id="admin-1", role="admin"),
    )

    assert calls == ["by_user", "by_id"]
    assert resolved == "pc:v1:admin-stored"


def test_injector_skips_non_responses_provider(monkeypatch):
    def fail_resolver(*_args, **_kwargs):
        raise AssertionError("resolver should not be called for non-responses providers")

    monkeypatch.setattr(
        openai_router, "_resolve_prompt_cache_key_for_responses", fail_resolver
    )

    payload = {"model": "gpt-5"}
    openai_router._inject_prompt_cache_key_for_responses(
        "openai",
        payload,
        {"chat_id": "chat-1"},
        _user(),
    )

    assert "prompt_cache_key" not in payload
