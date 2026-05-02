from types import SimpleNamespace

import pytest

from open_webui.utils import middleware


def _build_request(*, conversation_embedding_enabled: bool = False):
    config = SimpleNamespace(
        CONVERSATION_FILE_UPLOAD_EMBEDDING=conversation_embedding_enabled,
        RAG_TEMPLATE="RAG WRAP [context] QUERY [query]",
        RAG_FULL_CONTEXT=False,
        TOP_K_RERANKER=3,
        RELEVANCE_THRESHOLD=0.0,
        BM25_WEIGHT=0.5,
        ENABLE_RAG_BM25_SEARCH=False,
        ENABLE_RAG_RERANKING=False,
        ENABLE_RAG_BM25_ENRICHED_TEXTS=False,
        RETRIEVAL_CHUNK_EXPANSION=0,
        TASK_MODEL="",
    )
    state = SimpleNamespace(
        config=config,
        EMBEDDING_FUNCTION=lambda *_args, **_kwargs: [0.1, 0.2],
        RERANKING_FUNCTION=None,
        MODELS={"test-model": {"id": "test-model", "info": {"meta": {}}}},
    )
    return SimpleNamespace(
        app=SimpleNamespace(state=state),
        cookies={},
        state=SimpleNamespace(direct=False),
    )


def _build_file_obj(file_id: str, filename: str, content: str, meta: dict | None = None):
    return SimpleNamespace(
        id=file_id,
        filename=filename,
        data={"content": content},
        meta=meta or {},
    )


@pytest.mark.asyncio
async def test_knowledge_file_goes_through_retrieval_when_embedding_disabled(monkeypatch):
    request = _build_request(conversation_embedding_enabled=False)
    user = SimpleNamespace(id="user-1", role="admin")
    events = []
    calls = {"get_sources": 0}

    async def event_emitter(event):
        events.append(event)

    async def fake_generate_queries(*_args, **_kwargs):
        return {"choices": [{"message": {"content": '{"queries":["q"]}'}}]}

    async def fake_get_sources_from_items(**kwargs):
        calls["get_sources"] += 1
        return [
            {
                "source": {"id": "kb-file-1"},
                "document": ["retrieved chunk"],
                "metadata": [{"source": "kb.txt"}],
            }
        ]

    monkeypatch.setattr(
        middleware,
        "get_sources_from_items",
        fake_get_sources_from_items,
    )
    monkeypatch.setattr(middleware, "generate_queries", fake_generate_queries)
    monkeypatch.setattr(
        middleware.Files,
        "get_file_by_id",
        lambda _id: _build_file_obj(
            file_id="kb-file-1",
            filename="kb.txt",
            content="FULL FILE CONTENT",
            meta={"collection_name": "knowledge-1"},
        ),
    )

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "question"}],
        "metadata": {
            "files": [
                {
                    "type": "file",
                    "id": "kb-file-1",
                    "name": "kb.txt",
                    "content_type": "text/plain",
                }
            ]
        },
    }

    _, flags = await middleware.chat_completion_files_handler(
        request,
        body,
        {"__event_emitter__": event_emitter},
        user,
    )

    assert calls["get_sources"] == 1
    assert flags["sources"][0]["document"] == ["retrieved chunk"]
    assert flags["sources"][0]["document"] != ["FULL FILE CONTENT"]
    assert any(e["type"] == "status" for e in events)


@pytest.mark.asyncio
async def test_direct_context_file_still_uses_full_injection(monkeypatch):
    request = _build_request(conversation_embedding_enabled=False)
    user = SimpleNamespace(id="user-1", role="admin")
    events = []

    async def event_emitter(event):
        events.append(event)

    async def fail_get_sources_from_items(**_kwargs):
        raise AssertionError("retrieval should not run for direct_context files")

    monkeypatch.setattr(
        middleware,
        "get_sources_from_items",
        fail_get_sources_from_items,
    )
    monkeypatch.setattr(
        middleware.Files,
        "get_file_by_id",
        lambda _id: _build_file_obj(
            file_id="upload-1",
            filename="upload.txt",
            content="DIRECT CONTEXT CONTENT",
            meta={"data": {"conversation_ingest_mode": "direct_context"}},
        ),
    )

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "question"}],
        "metadata": {
            "files": [
                {
                    "type": "file",
                    "id": "upload-1",
                    "name": "upload.txt",
                    "content_type": "text/plain",
                    "ingest_mode": "direct_context",
                }
            ]
        },
    }

    _, flags = await middleware.chat_completion_files_handler(
        request,
        body,
        {"__event_emitter__": event_emitter},
        user,
    )

    assert flags["sources"] == []
    assert flags["context_sources"][0]["document"] == ["DIRECT CONTEXT CONTENT"]
    assert any(
        e.get("type") == "status"
        and e.get("data", {}).get("action") == "sources_retrieved"
        for e in events
    )


@pytest.mark.asyncio
async def test_direct_context_file_is_prepended_without_rag_template(monkeypatch):
    request = _build_request(conversation_embedding_enabled=False)
    user = SimpleNamespace(id="user-1", role="admin")
    events = []

    async def event_emitter(event):
        events.append(event)

    async def fail_get_sources_from_items(**_kwargs):
        raise AssertionError("retrieval should not run for direct_context files")

    monkeypatch.setattr(middleware, "get_event_emitter", lambda _metadata: event_emitter)
    monkeypatch.setattr(middleware, "get_event_call", lambda _metadata: None)
    monkeypatch.setattr(
        middleware,
        "get_sources_from_items",
        fail_get_sources_from_items,
    )
    monkeypatch.setattr(
        middleware.Files,
        "get_file_by_id",
        lambda _id: _build_file_obj(
            file_id="upload-1",
            filename="upload.txt",
            content="DIRECT CONTEXT CONTENT",
            meta={"data": {"conversation_ingest_mode": "direct_context"}},
        ),
    )

    form_data, _, _ = await middleware.process_chat_payload(
        request,
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "question"}],
            "files": [
                {
                    "type": "file",
                    "id": "upload-1",
                    "name": "upload.txt",
                    "content_type": "text/plain",
                    "ingest_mode": "direct_context",
                }
            ],
        },
        user,
        {},
        {"id": "test-model", "info": {"meta": {}}},
    )

    content = form_data["messages"][-1]["content"]
    assert content.startswith("Uploaded file contents:")
    assert "DIRECT CONTEXT CONTENT" in content
    assert content.index("DIRECT CONTEXT CONTENT") < content.index("question")
    assert "RAG WRAP" not in content
    assert not events or all(event.get("type") == "status" for event in events)


@pytest.mark.asyncio
async def test_direct_context_files_preserve_request_order(monkeypatch):
    request = _build_request(conversation_embedding_enabled=False)
    user = SimpleNamespace(id="user-1", role="admin")

    async def event_emitter(_event):
        return None

    files_by_id = {
        "file-1": _build_file_obj(
            "file-1",
            "first.txt",
            "FIRST CONTENT",
            {"data": {"conversation_ingest_mode": "direct_context"}},
        ),
        "file-2": _build_file_obj(
            "file-2",
            "second.txt",
            "SECOND CONTENT",
            {"data": {"conversation_ingest_mode": "direct_context"}},
        ),
    }

    monkeypatch.setattr(middleware, "get_event_emitter", lambda _metadata: event_emitter)
    monkeypatch.setattr(middleware, "get_event_call", lambda _metadata: None)
    monkeypatch.setattr(
        middleware,
        "get_sources_from_items",
        lambda **_kwargs: pytest.fail("retrieval should not run"),
    )
    monkeypatch.setattr(middleware.Files, "get_file_by_id", lambda file_id: files_by_id[file_id])

    form_data, _, _ = await middleware.process_chat_payload(
        request,
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "question"}],
            "files": [
                {
                    "type": "file",
                    "id": "file-1",
                    "name": "first.txt",
                    "content_type": "text/plain",
                    "ingest_mode": "direct_context",
                },
                {
                    "type": "file",
                    "id": "file-2",
                    "name": "second.txt",
                    "content_type": "text/plain",
                    "ingest_mode": "direct_context",
                },
            ],
        },
        user,
        {},
        {"id": "test-model", "info": {"meta": {}}},
    )

    content = form_data["messages"][-1]["content"]
    assert content.index("FIRST CONTENT") < content.index("SECOND CONTENT")
    assert content.index("SECOND CONTENT") < content.index("question")


@pytest.mark.asyncio
async def test_removed_direct_context_file_is_not_injected(monkeypatch):
    request = _build_request(conversation_embedding_enabled=False)
    user = SimpleNamespace(id="user-1", role="admin")

    async def event_emitter(_event):
        return None

    monkeypatch.setattr(middleware, "get_event_emitter", lambda _metadata: event_emitter)
    monkeypatch.setattr(middleware, "get_event_call", lambda _metadata: None)
    monkeypatch.setattr(
        middleware,
        "get_sources_from_items",
        lambda **_kwargs: pytest.fail("retrieval should not run"),
    )
    monkeypatch.setattr(
        middleware.Files,
        "get_file_by_id",
        lambda _id: _build_file_obj(
            file_id="file-2",
            filename="second.txt",
            content="SECOND CONTENT",
            meta={"data": {"conversation_ingest_mode": "direct_context"}},
        ),
    )

    form_data, _, _ = await middleware.process_chat_payload(
        request,
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "question"}],
            "files": [
                {
                    "type": "file",
                    "id": "file-2",
                    "name": "second.txt",
                    "content_type": "text/plain",
                    "ingest_mode": "direct_context",
                }
            ],
        },
        user,
        {},
        {"id": "test-model", "info": {"meta": {}}},
    )

    content = form_data["messages"][-1]["content"]
    assert "SECOND CONTENT" in content
    assert "FIRST CONTENT" not in content


@pytest.mark.asyncio
async def test_direct_context_file_and_rag_sources_are_injected_separately(monkeypatch):
    request = _build_request(conversation_embedding_enabled=False)
    user = SimpleNamespace(id="user-1", role="admin")

    async def event_emitter(_event):
        return None

    async def fake_generate_queries(*_args, **_kwargs):
        return {"choices": [{"message": {"content": '{"queries":["q"]}'}}]}

    async def fake_get_sources_from_items(**kwargs):
        assert [item["id"] for item in kwargs["items"]] == ["standard-1"]
        return [
            {
                "source": {"id": "knowledge-1", "name": "Knowledge"},
                "document": ["RAG CHUNK"],
                "metadata": [{"source": "kb.txt"}],
            }
        ]

    files_by_id = {
        "direct-1": _build_file_obj(
            "direct-1",
            "direct.txt",
            "DIRECT CONTEXT CONTENT",
            {"data": {"conversation_ingest_mode": "direct_context"}},
        ),
        "standard-1": _build_file_obj(
            "standard-1",
            "standard.txt",
            "STANDARD CONTENT",
            {"data": {"conversation_ingest_mode": "standard"}},
        ),
    }

    monkeypatch.setattr(middleware, "get_event_emitter", lambda _metadata: event_emitter)
    monkeypatch.setattr(middleware, "get_event_call", lambda _metadata: None)
    monkeypatch.setattr(middleware, "generate_queries", fake_generate_queries)
    monkeypatch.setattr(middleware, "get_sources_from_items", fake_get_sources_from_items)
    monkeypatch.setattr(middleware.Files, "get_file_by_id", lambda file_id: files_by_id[file_id])

    form_data, _, events = await middleware.process_chat_payload(
        request,
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "question"}],
            "files": [
                {
                    "type": "file",
                    "id": "direct-1",
                    "name": "direct.txt",
                    "content_type": "text/plain",
                    "ingest_mode": "direct_context",
                },
                {
                    "type": "file",
                    "id": "standard-1",
                    "name": "standard.txt",
                    "content_type": "text/plain",
                    "ingest_mode": "standard",
                },
            ],
        },
        user,
        {},
        {"id": "test-model", "info": {"meta": {}}},
    )

    content = form_data["messages"][-1]["content"]
    assert content.startswith("Uploaded file contents:")
    assert content.index("DIRECT CONTEXT CONTENT") < content.index("RAG WRAP")
    assert "RAG CHUNK" in content
    assert "<source" in content
    assert "DIRECT CONTEXT CONTENT</source>" not in content
    assert events == [
        {
            "sources": [
                {
                    "source": {"id": "knowledge-1", "name": "Knowledge"},
                    "document": ["RAG CHUNK"],
                    "metadata": [{"source": "kb.txt"}],
                }
            ]
        }
    ]


@pytest.mark.asyncio
async def test_image_file_does_not_use_direct_context_injection(monkeypatch):
    request = _build_request(conversation_embedding_enabled=False)
    user = SimpleNamespace(id="user-1", role="admin")
    calls = {"get_sources": 0}

    async def event_emitter(_event):
        return None

    async def fake_generate_queries(*_args, **_kwargs):
        return {"choices": [{"message": {"content": '{"queries":["q"]}'}}]}

    async def fake_get_sources_from_items(**kwargs):
        calls["get_sources"] += 1
        assert kwargs["items"][0]["content_type"] == "image/png"
        return []

    monkeypatch.setattr(middleware, "generate_queries", fake_generate_queries)
    monkeypatch.setattr(
        middleware,
        "get_sources_from_items",
        fake_get_sources_from_items,
    )

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "question"}],
        "metadata": {
            "files": [
                {
                    "type": "file",
                    "id": "img-1",
                    "name": "img.png",
                    "content_type": "image/png",
                }
            ]
        },
    }

    _, flags = await middleware.chat_completion_files_handler(
        request,
        body,
        {"__event_emitter__": event_emitter},
        user,
    )

    assert calls["get_sources"] == 1
    assert flags["sources"] == []


@pytest.mark.asyncio
async def test_full_context_file_keeps_full_context_retrieval_flow(monkeypatch):
    request = _build_request(conversation_embedding_enabled=False)
    user = SimpleNamespace(id="user-1", role="admin")
    captured = {}

    async def event_emitter(_event):
        return None

    async def fail_generate_queries(*_args, **_kwargs):
        raise AssertionError("generate_queries should not run when all files are full context")

    async def fake_get_sources_from_items(**kwargs):
        captured["full_context"] = kwargs["full_context"]
        return [{"source": {"id": "kb-file-2"}, "document": ["full"], "metadata": [{}]}]

    monkeypatch.setattr(middleware, "generate_queries", fail_generate_queries)
    monkeypatch.setattr(
        middleware,
        "get_sources_from_items",
        fake_get_sources_from_items,
    )
    monkeypatch.setattr(
        middleware.Files,
        "get_file_by_id",
        lambda _id: _build_file_obj(
            file_id="kb-file-2",
            filename="kb2.txt",
            content="FULL FILE CONTENT",
            meta={"collection_name": "knowledge-2"},
        ),
    )

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "question"}],
        "metadata": {
            "files": [
                {
                    "type": "file",
                    "id": "kb-file-2",
                    "name": "kb2.txt",
                    "content_type": "text/plain",
                    "context": "full",
                }
            ]
        },
    }

    _, flags = await middleware.chat_completion_files_handler(
        request,
        body,
        {"__event_emitter__": event_emitter},
        user,
    )

    assert captured["full_context"] is True
    assert flags["sources"][0]["document"] == ["full"]
