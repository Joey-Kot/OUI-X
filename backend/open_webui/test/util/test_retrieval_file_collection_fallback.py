from types import SimpleNamespace

import pytest

from open_webui.retrieval import utils as retrieval_utils


def _request_config():
    return SimpleNamespace(
        BYPASS_EMBEDDING_AND_RETRIEVAL=False,
        CONVERSATION_FILE_UPLOAD_EMBEDDING=True,
        TOP_K=4,
    )


@pytest.mark.asyncio
async def test_file_scope_uses_item_meta_collection_name(monkeypatch):
    user = SimpleNamespace(id="u-1", role="admin")
    captured = {}

    def fake_runtime(_request, collection_name):
        captured["runtime_collection_name"] = collection_name
        return {
            "physical_collection_name": "kb-physical",
            "effective_config": {
                "TOP_K": 3,
                "TOP_K_RERANKER": 2,
                "RELEVANCE_THRESHOLD": 0.0,
                "ENABLE_RAG_RERANKING": False,
                "RETRIEVAL_CHUNK_EXPANSION": 0,
            },
            "embedding_function": lambda *_args, **_kwargs: [0.1, 0.2],
            "reranking_function": None,
        }

    async def fake_query_doc_with_file_scope(**kwargs):
        captured["query_collection_name"] = kwargs["collection_name"]
        return {
            "documents": [["chunk"]],
            "metadatas": [[{"source": "kb.txt"}]],
            "distances": [[0.1]],
        }

    monkeypatch.setattr(retrieval_utils, "_build_collection_runtime_functions", fake_runtime)
    monkeypatch.setattr(
        retrieval_utils,
        "query_doc_with_file_scope",
        fake_query_doc_with_file_scope,
    )
    monkeypatch.setattr(retrieval_utils.Files, "get_file_by_id", lambda _id: None)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(config=_request_config())))
    items = [
        {
            "type": "file",
            "id": "file-1",
            "name": "kb.txt",
            "meta": {"collection_name": "knowledge-from-item-meta"},
        }
    ]

    sources = await retrieval_utils.get_sources_from_items(
        request=request,
        items=items,
        queries=["hello"],
        embedding_function=lambda *_args, **_kwargs: [0.1, 0.2],
        k=None,
        reranking_function=None,
        k_reranker=2,
        r=0.0,
        bm25_weight=0.5,
        enable_bm25_search=False,
        enable_reranking=False,
        enable_bm25_enriched_texts=False,
        retrieval_chunk_expansion=0,
        full_context=False,
        user=user,
    )

    assert captured["runtime_collection_name"] == "knowledge-from-item-meta"
    assert captured["query_collection_name"] == "kb-physical"
    assert sources[0]["document"] == ["chunk"]


@pytest.mark.asyncio
async def test_file_scope_uses_file_db_meta_collection_name(monkeypatch):
    user = SimpleNamespace(id="u-1", role="admin")
    captured = {}

    def fake_runtime(_request, collection_name):
        captured["runtime_collection_name"] = collection_name
        return {
            "physical_collection_name": "kb-physical-db",
            "effective_config": {
                "TOP_K": 3,
                "TOP_K_RERANKER": 2,
                "RELEVANCE_THRESHOLD": 0.0,
                "ENABLE_RAG_RERANKING": False,
                "RETRIEVAL_CHUNK_EXPANSION": 0,
            },
            "embedding_function": lambda *_args, **_kwargs: [0.1, 0.2],
            "reranking_function": None,
        }

    async def fake_query_doc_with_file_scope(**kwargs):
        captured["query_collection_name"] = kwargs["collection_name"]
        return {
            "documents": [["chunk-db"]],
            "metadatas": [[{"source": "db.txt"}]],
            "distances": [[0.2]],
        }

    monkeypatch.setattr(retrieval_utils, "_build_collection_runtime_functions", fake_runtime)
    monkeypatch.setattr(
        retrieval_utils,
        "query_doc_with_file_scope",
        fake_query_doc_with_file_scope,
    )
    monkeypatch.setattr(
        retrieval_utils.Files,
        "get_file_by_id",
        lambda _id: SimpleNamespace(
            meta={"collection_name": "knowledge-from-db-meta"},
            data={"content": "db-content"},
            filename="db.txt",
        ),
    )

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(config=_request_config())))
    items = [{"type": "file", "id": "file-2", "name": "db.txt"}]

    sources = await retrieval_utils.get_sources_from_items(
        request=request,
        items=items,
        queries=["hello"],
        embedding_function=lambda *_args, **_kwargs: [0.1, 0.2],
        k=None,
        reranking_function=None,
        k_reranker=2,
        r=0.0,
        bm25_weight=0.5,
        enable_bm25_search=False,
        enable_reranking=False,
        enable_bm25_enriched_texts=False,
        retrieval_chunk_expansion=0,
        full_context=False,
        user=user,
    )

    assert captured["runtime_collection_name"] == "knowledge-from-db-meta"
    assert captured["query_collection_name"] == "kb-physical-db"
    assert sources[0]["document"] == ["chunk-db"]
