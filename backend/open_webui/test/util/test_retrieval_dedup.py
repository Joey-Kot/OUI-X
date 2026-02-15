import pytest
from langchain_core.documents import Document

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
