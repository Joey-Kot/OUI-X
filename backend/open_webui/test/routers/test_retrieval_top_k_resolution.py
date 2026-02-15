from types import SimpleNamespace

import pytest

from open_webui.routers import retrieval as retrieval_router


def _make_request(top_k: int = 25):
    config = SimpleNamespace(TOP_K=top_k)
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(config=config)))


def _make_effective_config(top_k: int) -> dict:
    return {
        "TOP_K": top_k,
        "TOP_K_RERANKER": 3,
        "RELEVANCE_THRESHOLD": 0.0,
        "BM25_WEIGHT": 0.5,
        "ENABLE_RAG_BM25_SEARCH": False,
        "ENABLE_RAG_BM25_ENRICHED_TEXTS": False,
        "ENABLE_RAG_RERANKING": False,
        "RETRIEVAL_CHUNK_EXPANSION": 0,
        "RAG_EMBEDDING_ENGINE": "openai",
        "RAG_EMBEDDING_MODEL": "test-embedding",
        "RAG_RERANKING_ENGINE": "external",
        "RAG_RERANKING_MODEL": "",
        "RAG_EMBEDDING_BATCH_SIZE": 1,
    }


@pytest.mark.asyncio
async def test_query_doc_handler_honors_explicit_zero_k(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        retrieval_router, "get_physical_collection_name", lambda collection_name: collection_name
    )
    monkeypatch.setattr(
        retrieval_router,
        "get_collection_effective_config",
        lambda _request, _collection_name: _make_effective_config(5),
    )

    async def fake_embedding_function(query, prefix, user=None):
        del query, prefix, user
        return [0.1, 0.2]

    monkeypatch.setattr(
        retrieval_router,
        "build_embedding_function_from_effective_config",
        lambda _request, _effective_config: fake_embedding_function,
    )
    monkeypatch.setattr(
        retrieval_router,
        "build_reranking_function_from_effective_config",
        lambda _request, _effective_config: None,
    )

    async def fake_query_doc_with_rag_pipeline(**kwargs):
        captured["k"] = kwargs["k"]
        return {
            "distances": [[]],
            "documents": [[]],
            "metadatas": [[]],
        }

    monkeypatch.setattr(
        retrieval_router, "query_doc_with_rag_pipeline", fake_query_doc_with_rag_pipeline
    )

    request = _make_request(top_k=25)
    user = SimpleNamespace(id="u-1", role="admin")
    form_data = retrieval_router.QueryDocForm(
        collection_name="kb-1",
        query="hello",
        k=0,
    )

    await retrieval_router.query_doc_handler(request, form_data, user)

    assert captured["k"] == 0


@pytest.mark.asyncio
async def test_query_collection_handler_uses_per_collection_top_k_and_max_merge(
    monkeypatch,
):
    captured = {"ks": [], "merge_k": None}

    monkeypatch.setattr(
        retrieval_router, "get_physical_collection_name", lambda collection_name: collection_name
    )

    def fake_get_collection_effective_config(_request, collection_name):
        if collection_name == "kb-a":
            return _make_effective_config(4)
        return _make_effective_config(9)

    monkeypatch.setattr(
        retrieval_router,
        "get_collection_effective_config",
        fake_get_collection_effective_config,
    )

    async def fake_embedding_function(query, prefix, user=None):
        del query, prefix, user
        return [0.1, 0.2]

    monkeypatch.setattr(
        retrieval_router,
        "build_embedding_function_from_effective_config",
        lambda _request, _effective_config: fake_embedding_function,
    )
    monkeypatch.setattr(
        retrieval_router,
        "build_reranking_function_from_effective_config",
        lambda _request, _effective_config: None,
    )

    async def fake_query_doc_with_rag_pipeline(**kwargs):
        captured["ks"].append(kwargs["k"])
        return {
            "distances": [[1.0]],
            "documents": [["chunk"]],
            "metadatas": [[{"source": "s"}]],
        }

    monkeypatch.setattr(
        retrieval_router, "query_doc_with_rag_pipeline", fake_query_doc_with_rag_pipeline
    )
    monkeypatch.setattr(
        retrieval_router,
        "merge_and_sort_query_results",
        lambda _results, k: captured.update({"merge_k": k}) or {"k": k},
    )

    request = _make_request(top_k=25)
    user = SimpleNamespace(id="u-1", role="admin")
    form_data = retrieval_router.QueryCollectionsForm(
        collection_names=["kb-a", "kb-b"],
        query="hello",
    )

    result = await retrieval_router.query_collection_handler(request, form_data, user)

    assert captured["ks"] == [4, 9]
    assert captured["merge_k"] == 9
    assert result == {"k": 9}


@pytest.mark.asyncio
async def test_query_collection_handler_explicit_k_overrides_collection_top_k(monkeypatch):
    captured = {"ks": [], "merge_k": None}

    monkeypatch.setattr(
        retrieval_router, "get_physical_collection_name", lambda collection_name: collection_name
    )
    monkeypatch.setattr(
        retrieval_router,
        "get_collection_effective_config",
        lambda _request, _collection_name: _make_effective_config(5),
    )

    async def fake_embedding_function(query, prefix, user=None):
        del query, prefix, user
        return [0.1, 0.2]

    monkeypatch.setattr(
        retrieval_router,
        "build_embedding_function_from_effective_config",
        lambda _request, _effective_config: fake_embedding_function,
    )
    monkeypatch.setattr(
        retrieval_router,
        "build_reranking_function_from_effective_config",
        lambda _request, _effective_config: None,
    )

    async def fake_query_doc_with_rag_pipeline(**kwargs):
        captured["ks"].append(kwargs["k"])
        return {
            "distances": [[1.0]],
            "documents": [["chunk"]],
            "metadatas": [[{"source": "s"}]],
        }

    monkeypatch.setattr(
        retrieval_router, "query_doc_with_rag_pipeline", fake_query_doc_with_rag_pipeline
    )
    monkeypatch.setattr(
        retrieval_router,
        "merge_and_sort_query_results",
        lambda _results, k: captured.update({"merge_k": k}) or {"k": k},
    )

    request = _make_request(top_k=25)
    user = SimpleNamespace(id="u-1", role="admin")
    form_data = retrieval_router.QueryCollectionsForm(
        collection_names=["kb-a", "kb-b"],
        query="hello",
        k=12,
    )

    result = await retrieval_router.query_collection_handler(request, form_data, user)

    assert captured["ks"] == [12, 12]
    assert captured["merge_k"] == 12
    assert result == {"k": 12}
