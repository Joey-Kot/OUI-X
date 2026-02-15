import pytest
from langchain_core.documents import Document
from types import SimpleNamespace

from open_webui.retrieval import utils as retrieval_utils


def test_dedupe_documents_metadata_key_prefers_file_and_start_index():
    docs = [
        Document(
            page_content="first chunk",
            metadata={"file_id": "file-1", "start_index": 0},
        ),
        Document(
            page_content="first chunk duplicate text changed",
            metadata={"file_id": "file-1", "start_index": 0},
        ),
        Document(
            page_content="second chunk",
            metadata={"file_id": "file-1", "start_index": 10},
        ),
    ]

    deduped = retrieval_utils.dedupe_documents_before_rerank(docs)

    assert len(deduped) == 2
    assert deduped[0].page_content == "first chunk"
    assert deduped[1].page_content == "second chunk"


def test_dedupe_documents_falls_back_to_normalized_text_hash():
    docs = [
        Document(page_content="  hello   world\n", metadata={}),
        Document(page_content="hello world", metadata={}),
        Document(page_content="hello world!", metadata={}),
    ]

    deduped = retrieval_utils.dedupe_documents_before_rerank(docs)

    assert len(deduped) == 2
    assert [doc.page_content for doc in deduped] == ["  hello   world\n", "hello world!"]


def test_dedupe_documents_does_not_merge_same_text_with_different_start_index():
    docs = [
        Document(
            page_content="same sentence",
            metadata={"source": "doc-a", "start_index": 0},
        ),
        Document(
            page_content="same sentence",
            metadata={"source": "doc-a", "start_index": 50},
        ),
    ]

    deduped = retrieval_utils.dedupe_documents_before_rerank(docs)

    assert len(deduped) == 2


def test_dedupe_documents_preserves_order_of_first_occurrence():
    docs = [
        Document(page_content="A", metadata={"file_id": "f", "start_index": 0}),
        Document(page_content="B", metadata={"file_id": "f", "start_index": 10}),
        Document(page_content="A-duplicate", metadata={"file_id": "f", "start_index": 0}),
        Document(page_content="C", metadata={"file_id": "f", "start_index": 20}),
    ]

    deduped = retrieval_utils.dedupe_documents_before_rerank(docs)

    assert [doc.page_content for doc in deduped] == ["A", "B", "C"]


class _FakeSearchResult:
    def __init__(self, ids, metadatas, documents):
        self.ids = [ids]
        self.metadatas = [metadatas]
        self.documents = [documents]


class _FakeGetResult:
    def __init__(self, ids, metadatas, documents):
        self.ids = [ids]
        self.metadatas = [metadatas]
        self.documents = [documents]


class _FakeVectorClient:
    def search(self, collection_name, vectors, limit):
        del collection_name, vectors, limit
        return _FakeSearchResult(
            ids=["1", "2", "3"],
            metadatas=[
                {"file_id": "file-1", "start_index": 0},
                {"file_id": "file-1", "start_index": 0},
                {"file_id": "file-1", "start_index": 100},
            ],
            documents=["alpha", "alpha duplicate", "beta"],
        )


class _FakeVectorClientWithExpansion:
    def __init__(self):
        self._all_docs = _FakeGetResult(
            ids=["0", "10", "20", "30"],
            metadatas=[
                {"file_id": "file-1", "start_index": 0},
                {"file_id": "file-1", "start_index": 10},
                {"file_id": "file-1", "start_index": 20},
                {"file_id": "file-1", "start_index": 30},
            ],
            documents=["A", "B", "C", "D"],
        )

    def search(self, collection_name, vectors, limit):
        del collection_name, vectors, limit
        return _FakeSearchResult(
            ids=["10", "20"],
            metadatas=[
                {"file_id": "file-1", "start_index": 10},
                {"file_id": "file-1", "start_index": 20},
            ],
            documents=["B", "C"],
        )

    def get(self, collection_name):
        del collection_name
        return self._all_docs

    def query(self, collection_name, filter, limit=None):
        del collection_name, limit
        if filter == {"file_id": "file-1"}:
            return self._all_docs
        return _FakeGetResult(ids=[], metadatas=[], documents=[])


class _FakeFileScopeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.k = 0

    async def ainvoke(self, query):
        del query
        return [self.docs[1]]


class _FakeBM25Retriever:
    @classmethod
    def from_documents(cls, docs):
        return _FakeFileScopeRetriever(docs)


@pytest.mark.asyncio
async def test_query_doc_with_rag_pipeline_dedupes_before_rerank(monkeypatch):
    monkeypatch.setattr(retrieval_utils, "VECTOR_DB_CLIENT", _FakeVectorClient())
    rerank_input_counts = []

    async def embedding_function(_query, _prefix):
        return [0.1, 0.2]

    def reranking_function(_query, documents):
        rerank_input_counts.append(len(documents))
        # Keep all docs and stable order.
        return [1.0 - i * 0.1 for i in range(len(documents))]

    result = await retrieval_utils.query_doc_with_rag_pipeline(
        collection_name="test-collection",
        query="query",
        embedding_function=embedding_function,
        k=10,
        reranking_function=reranking_function,
        k_reranker=10,
        r=0.0,
        bm25_weight=0.5,
        enable_bm25_search=False,
        enable_reranking=True,
        collection_result=None,
        enable_bm25_enriched_texts=False,
    )

    assert rerank_input_counts == [2]
    assert result["documents"][0] == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_query_doc_with_rag_pipeline_expands_neighbors_after_rerank(monkeypatch):
    monkeypatch.setattr(
        retrieval_utils, "VECTOR_DB_CLIENT", _FakeVectorClientWithExpansion()
    )
    rerank_inputs = []

    async def embedding_function(_query, _prefix):
        return [0.1, 0.2]

    def reranking_function(_query, documents):
        rerank_inputs.append([doc.page_content for doc in documents])
        return [1.0 - i * 0.1 for i in range(len(documents))]

    result = await retrieval_utils.query_doc_with_rag_pipeline(
        collection_name="test-collection",
        query="query",
        embedding_function=embedding_function,
        k=10,
        reranking_function=reranking_function,
        k_reranker=10,
        r=0.0,
        bm25_weight=0.5,
        enable_bm25_search=False,
        enable_reranking=True,
        collection_result=None,
        enable_bm25_enriched_texts=False,
        retrieval_chunk_expansion=1,
    )

    assert rerank_inputs == [["B", "C"]]
    assert result["documents"][0] == ["B", "A", "C", "D"]
    assert result["metadatas"][0][1]["retrieval_chunk_expanded"] is True
    assert result["metadatas"][0][3]["retrieval_chunk_expanded"] is True
    assert result["distances"][0][1] == 1.0
    assert result["distances"][0][3] == 0.9


@pytest.mark.asyncio
async def test_query_doc_with_file_scope_expands_neighbors(monkeypatch):
    monkeypatch.setattr(
        retrieval_utils, "VECTOR_DB_CLIENT", _FakeVectorClientWithExpansion()
    )
    monkeypatch.setattr(retrieval_utils, "BM25Retriever", _FakeBM25Retriever)

    async def embedding_function(_query, _prefix):
        return [0.1, 0.2]

    result = await retrieval_utils.query_doc_with_file_scope(
        collection_name="test-collection",
        file_id="file-1",
        query="query",
        embedding_function=embedding_function,
        k=10,
        reranking_function=None,
        k_reranker=10,
        r=0.0,
        enable_reranking=False,
        retrieval_chunk_expansion=1,
    )

    assert result is not None
    assert result["documents"][0] == ["B", "A", "C"]
    assert result["metadatas"][0][1]["retrieval_chunk_expanded"] is True
    assert result["metadatas"][0][2]["retrieval_chunk_expanded"] is True
    assert result["distances"][0][1] is None
    assert result["distances"][0][2] is None


@pytest.mark.asyncio
async def test_query_doc_with_rag_pipeline_dedupes_when_rerank_disabled(monkeypatch):
    monkeypatch.setattr(retrieval_utils, "VECTOR_DB_CLIENT", _FakeVectorClient())

    async def embedding_function(_query, _prefix):
        return [0.1, 0.2]

    result = await retrieval_utils.query_doc_with_rag_pipeline(
        collection_name="test-collection",
        query="query",
        embedding_function=embedding_function,
        k=10,
        reranking_function=None,
        k_reranker=10,
        r=0.0,
        bm25_weight=0.5,
        enable_bm25_search=False,
        enable_reranking=False,
        collection_result=None,
        enable_bm25_enriched_texts=False,
    )

    assert result["documents"][0] == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_get_sources_from_items_uses_collection_top_k_when_k_not_explicit(
    monkeypatch,
):
    k_calls = []

    async def fake_collection_embedding(_query, prefix, user=None):
        del prefix, user
        return [0.1, 0.2]

    def fake_runtime(_request, collection_name):
        assert collection_name == "kb-1"
        return {
            "physical_collection_name": "phys-kb-1",
            "effective_config": {
                "TOP_K": 5,
                "TOP_K_RERANKER": 3,
                "RELEVANCE_THRESHOLD": 0.0,
                "BM25_WEIGHT": 0.5,
                "ENABLE_RAG_BM25_SEARCH": False,
                "ENABLE_RAG_BM25_ENRICHED_TEXTS": False,
                "ENABLE_RAG_RERANKING": False,
                "RETRIEVAL_CHUNK_EXPANSION": 0,
            },
            "embedding_function": fake_collection_embedding,
            "reranking_function": None,
        }

    async def fake_query_doc_with_rag_pipeline(*args, **kwargs):
        del args
        k_calls.append(kwargs["k"])
        return {
            "distances": [[1.0]],
            "documents": [["chunk"]],
            "metadatas": [[{"file_id": "f-1"}]],
        }

    monkeypatch.setattr(retrieval_utils, "_build_collection_runtime_functions", fake_runtime)
    monkeypatch.setattr(
        retrieval_utils,
        "query_doc_with_rag_pipeline",
        fake_query_doc_with_rag_pipeline,
    )
    monkeypatch.setattr(
        retrieval_utils.Knowledges,
        "get_knowledge_by_id",
        lambda _id: SimpleNamespace(id=_id, user_id="u-1", access_control=None),
    )

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(config=SimpleNamespace(TOP_K=25)))
    )
    user = SimpleNamespace(id="u-1", role="admin")

    sources = await retrieval_utils.get_sources_from_items(
        request=request,
        items=[{"type": "collection", "id": "kb-1"}],
        queries=["hello"],
        embedding_function=lambda *_args, **_kwargs: [0.3, 0.4],
        k=None,
        reranking_function=None,
        k_reranker=3,
        r=0.0,
        bm25_weight=0.5,
        enable_bm25_search=False,
        enable_reranking=False,
        enable_bm25_enriched_texts=False,
        retrieval_chunk_expansion=0,
        full_context=False,
        user=user,
    )

    assert k_calls == [5]
    assert len(sources) == 1


@pytest.mark.asyncio
async def test_get_sources_from_items_explicit_k_overrides_collection_top_k(monkeypatch):
    k_calls = []

    async def fake_collection_embedding(_query, prefix, user=None):
        del prefix, user
        return [0.1, 0.2]

    def fake_runtime(_request, collection_name):
        assert collection_name == "kb-1"
        return {
            "physical_collection_name": "phys-kb-1",
            "effective_config": {
                "TOP_K": 5,
                "TOP_K_RERANKER": 3,
                "RELEVANCE_THRESHOLD": 0.0,
                "BM25_WEIGHT": 0.5,
                "ENABLE_RAG_BM25_SEARCH": False,
                "ENABLE_RAG_BM25_ENRICHED_TEXTS": False,
                "ENABLE_RAG_RERANKING": False,
                "RETRIEVAL_CHUNK_EXPANSION": 0,
            },
            "embedding_function": fake_collection_embedding,
            "reranking_function": None,
        }

    async def fake_query_doc_with_rag_pipeline(*args, **kwargs):
        del args
        k_calls.append(kwargs["k"])
        return {
            "distances": [[1.0]],
            "documents": [["chunk"]],
            "metadatas": [[{"file_id": "f-1"}]],
        }

    monkeypatch.setattr(retrieval_utils, "_build_collection_runtime_functions", fake_runtime)
    monkeypatch.setattr(
        retrieval_utils,
        "query_doc_with_rag_pipeline",
        fake_query_doc_with_rag_pipeline,
    )
    monkeypatch.setattr(
        retrieval_utils.Knowledges,
        "get_knowledge_by_id",
        lambda _id: SimpleNamespace(id=_id, user_id="u-1", access_control=None),
    )

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(config=SimpleNamespace(TOP_K=25)))
    )
    user = SimpleNamespace(id="u-1", role="admin")

    await retrieval_utils.get_sources_from_items(
        request=request,
        items=[{"type": "collection", "id": "kb-1"}],
        queries=["hello"],
        embedding_function=lambda *_args, **_kwargs: [0.3, 0.4],
        k=12,
        reranking_function=None,
        k_reranker=3,
        r=0.0,
        bm25_weight=0.5,
        enable_bm25_search=False,
        enable_reranking=False,
        enable_bm25_enriched_texts=False,
        retrieval_chunk_expansion=0,
        full_context=False,
        user=user,
    )

    assert k_calls == [12]


@pytest.mark.asyncio
async def test_get_sources_from_items_uses_max_merge_k_for_multiple_collections(
    monkeypatch,
):
    captured = {"ks": [], "merge_k": None}

    async def fake_collection_embedding(_query, prefix, user=None):
        del prefix, user
        return [0.1, 0.2]

    def fake_runtime(_request, collection_name):
        top_k = 4 if collection_name == "kb-a" else 9
        return {
            "physical_collection_name": f"phys-{collection_name}",
            "effective_config": {
                "TOP_K": top_k,
                "TOP_K_RERANKER": 3,
                "RELEVANCE_THRESHOLD": 0.0,
                "BM25_WEIGHT": 0.5,
                "ENABLE_RAG_BM25_SEARCH": False,
                "ENABLE_RAG_BM25_ENRICHED_TEXTS": False,
                "ENABLE_RAG_RERANKING": False,
                "RETRIEVAL_CHUNK_EXPANSION": 0,
            },
            "embedding_function": fake_collection_embedding,
            "reranking_function": None,
        }

    async def fake_query_doc_with_rag_pipeline(*args, **kwargs):
        del args
        captured["ks"].append(kwargs["k"])
        return {
            "distances": [[1.0]],
            "documents": [[f"chunk-{kwargs['collection_name']}"]],
            "metadatas": [[{"file_id": "f-1"}]],
        }

    def fake_merge_and_sort_query_results(results, k):
        captured["merge_k"] = k
        merged_documents = []
        merged_metadatas = []
        merged_distances = []
        for item in results:
            merged_documents.extend(item["documents"][0])
            merged_metadatas.extend(item["metadatas"][0])
            merged_distances.extend(item["distances"][0])
        return {
            "distances": [merged_distances[:k]],
            "documents": [merged_documents[:k]],
            "metadatas": [merged_metadatas[:k]],
        }

    monkeypatch.setattr(retrieval_utils, "_build_collection_runtime_functions", fake_runtime)
    monkeypatch.setattr(
        retrieval_utils,
        "query_doc_with_rag_pipeline",
        fake_query_doc_with_rag_pipeline,
    )
    monkeypatch.setattr(
        retrieval_utils,
        "merge_and_sort_query_results",
        fake_merge_and_sort_query_results,
    )
    monkeypatch.setattr(retrieval_utils.Knowledges, "get_knowledge_by_id", lambda _id: None)

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(config=SimpleNamespace(TOP_K=25)))
    )
    user = SimpleNamespace(id="u-1", role="admin")

    await retrieval_utils.get_sources_from_items(
        request=request,
        items=[{"collection_names": ["kb-a", "kb-b"]}],
        queries=["hello"],
        embedding_function=lambda *_args, **_kwargs: [0.3, 0.4],
        k=None,
        reranking_function=None,
        k_reranker=3,
        r=0.0,
        bm25_weight=0.5,
        enable_bm25_search=False,
        enable_reranking=False,
        enable_bm25_enriched_texts=False,
        retrieval_chunk_expansion=0,
        full_context=False,
        user=user,
    )

    assert captured["ks"] == [4, 9]
    assert captured["merge_k"] == 9
