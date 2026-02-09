import asyncio
from types import SimpleNamespace

import pytest

from open_webui.models.files import FileModel
from open_webui.routers import knowledge as knowledge_router


class FakeChromaCollection:
    def __init__(self, payload=None):
        self.payload = payload or {}
        self.upserts = []
        self.calls = []

    def get(self, include=None, limit=None, offset=None, ids=None, where=None):
        self.calls.append(
            {
                "include": include,
                "limit": limit,
                "offset": offset,
                "ids": ids,
                "where": where,
            }
        )

        if callable(self.payload):
            return self.payload(include=include, limit=limit, offset=offset, ids=ids, where=where)

        return self.payload

    def count(self):
        if callable(self.payload):
            return self.payload(count_only=True)

        ids = self.payload.get("ids")
        if ids is None:
            return 0

        tolist = getattr(ids, "tolist", None)
        if callable(tolist):
            ids = tolist()

        return len(ids)

    def upsert(self, ids, embeddings, documents, metadatas):
        self.upserts.append(
            {
                "ids": ids,
                "embeddings": embeddings,
                "documents": documents,
                "metadatas": metadatas,
            }
        )


class FakeChromaClient:
    def __init__(self, source_payload):
        self.source_payload = source_payload
        self.collections = {}
        self.source_collection = FakeChromaCollection(payload=self.source_payload)

    def get_collection(self, name):
        if name != "collection-source":
            raise RuntimeError("missing")
        return self.source_collection

    def get_or_create_collection(self, name, metadata=None):
        collection = self.collections.get(name)
        if collection is None:
            collection = FakeChromaCollection()
            self.collections[name] = collection
        return collection


class FakeVectorClient:
    def __init__(self, client, existing_collections=None):
        self.client = client
        self.existing_collections = set(existing_collections or set())
        self.deleted_collections = []

    def has_collection(self, collection_name):
        return collection_name in self.existing_collections

    def delete_collection(self, collection_name):
        self.deleted_collections.append(collection_name)
        self.existing_collections.discard(collection_name)


class FakeBatchResult:
    def __init__(self, file_ids):
        self.results = [
            SimpleNamespace(file_id=file_id, status="completed", error=None)
            for file_id in file_ids
        ]
        self.errors = []


class AmbiguousArray:
    def __init__(self, data):
        self._data = data

    def __bool__(self):
        raise ValueError(
            "The truth value of an array with more than one element is ambiguous"
        )

    def tolist(self):
        return self._data


def _file(file_id: str) -> FileModel:
    return FileModel(
        id=file_id,
        user_id="u1",
        hash=None,
        filename=f"{file_id}.txt",
        path=None,
        data={"content": "test"},
        meta={"file_id": file_id},
        access_control=None,
        created_at=0,
        updated_at=0,
    )


def test_clone_collection_vectors_chroma_direct_copy(monkeypatch):
    files = [_file("file-1"), _file("file-2")]

    source_payload = {
        "ids": ["chunk-1", "chunk-2"],
        "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        "documents": ["doc1", "doc2"],
        "metadatas": [{"file_id": "file-1"}, {"file_id": "file-2"}],
    }

    fake_client = FakeVectorClient(
        client=FakeChromaClient(source_payload),
        existing_collections={"collection-source"},
    )

    process_calls = []

    async def fake_process_files_batch(request, form_data, user):
        process_calls.append(form_data)
        return FakeBatchResult([])

    monkeypatch.setattr(knowledge_router, "VECTOR_DB", "chroma")
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", fake_client)
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda knowledge_id, meta: f"collection-{knowledge_id}",
    )
    monkeypatch.setattr(knowledge_router, "process_files_batch", fake_process_files_batch)

    result = asyncio.run(
        knowledge_router._clone_collection_vectors(
            request=SimpleNamespace(),
            source_knowledge=SimpleNamespace(id="source", meta={}),
            target_knowledge=SimpleNamespace(id="target", meta={}),
            files=files,
            user=SimpleNamespace(id="u1"),
        )
    )

    assert result["successful_file_ids"] == ["file-1", "file-2"]
    assert result["warnings"]["strategy"] == "direct_copy"
    assert result["warnings"]["copied_count"] == 2
    assert result["warnings"]["total_items"] == 2
    assert result["warnings"]["pages_scanned"] == 1
    assert result["warnings"]["failed_offset"] is None
    assert process_calls == []


def test_clone_collection_vectors_chroma_partial_copy_then_reembed(monkeypatch):
    files = [_file("file-1"), _file("file-2")]

    source_payload = {
        "ids": ["chunk-1", "chunk-2"],
        "embeddings": [[0.1, 0.2], None],
        "documents": ["doc1", "doc2"],
        "metadatas": [{"file_id": "file-1"}, {"file_id": "file-2"}],
    }

    fake_client = FakeVectorClient(
        client=FakeChromaClient(source_payload),
        existing_collections={"collection-source"},
    )

    process_calls = []

    async def fake_process_files_batch(request, form_data, user):
        process_calls.append([file.id for file in form_data.files])
        return FakeBatchResult(["file-2"])

    monkeypatch.setattr(knowledge_router, "VECTOR_DB", "chroma")
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", fake_client)
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda knowledge_id, meta: f"collection-{knowledge_id}",
    )
    monkeypatch.setattr(knowledge_router, "process_files_batch", fake_process_files_batch)

    result = asyncio.run(
        knowledge_router._clone_collection_vectors(
            request=SimpleNamespace(),
            source_knowledge=SimpleNamespace(id="source", meta={}),
            target_knowledge=SimpleNamespace(id="target", meta={}),
            files=files,
            user=SimpleNamespace(id="u1"),
        )
    )

    assert sorted(result["successful_file_ids"]) == ["file-1", "file-2"]
    assert result["warnings"]["strategy"] == "partial_copy_with_reembed"
    assert result["warnings"]["copied_count"] == 1
    assert result["warnings"]["reembedded_file_ids"] == ["file-2"]
    assert "incomplete embeddings" in result["warnings"]["reason"]
    assert result["warnings"]["missing_chunks"] == 1
    assert result["warnings"]["failed_stage"] is None
    assert process_calls == [["file-2"]]


def test_clone_collection_vectors_fallback_reembeds_all_files(monkeypatch):
    files = [_file("file-1"), _file("file-2")]

    fake_client = FakeVectorClient(
        client=FakeChromaClient({}),
        existing_collections={"collection-target"},
    )

    process_calls = []

    async def fake_process_files_batch(request, form_data, user):
        process_calls.append([file.id for file in form_data.files])
        return FakeBatchResult(["file-1", "file-2"])

    monkeypatch.setattr(knowledge_router, "VECTOR_DB", "chroma")
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", fake_client)
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda knowledge_id, meta: f"collection-{knowledge_id}",
    )
    monkeypatch.setattr(knowledge_router, "process_files_batch", fake_process_files_batch)

    result = asyncio.run(
        knowledge_router._clone_collection_vectors(
            request=SimpleNamespace(),
            source_knowledge=SimpleNamespace(id="source", meta={}),
            target_knowledge=SimpleNamespace(id="target", meta={}),
            files=files,
            user=SimpleNamespace(id="u1"),
        )
    )

    assert sorted(result["successful_file_ids"]) == ["file-1", "file-2"]
    assert result["warnings"]["strategy"] == "full_reembed_fallback"
    assert result["warnings"]["reason"] == "source vector collection does not exist"
    assert result["warnings"]["reembedded_file_ids"] == ["file-1", "file-2"]
    assert result["warnings"]["failed_stage"] is None
    assert result["warnings"]["failed_offset"] is None
    assert process_calls == [["file-1", "file-2"]]
    assert fake_client.deleted_collections == ["collection-target"]


def test_chroma_has_collection_supports_multiple_list_shapes():
    pytest.importorskip("chromadb")

    from open_webui.retrieval.vector.dbs.chroma import ChromaClient

    class CollectionObject:
        def __init__(self, name):
            self.name = name

    fake_self = SimpleNamespace(
        client=SimpleNamespace(
            list_collections=lambda: [
                "string-collection",
                CollectionObject("object-collection"),
                {"name": "dict-collection"},
            ]
        )
    )

    assert ChromaClient.has_collection(fake_self, "string-collection") is True
    assert ChromaClient.has_collection(fake_self, "object-collection") is True
    assert ChromaClient.has_collection(fake_self, "dict-collection") is True
    assert ChromaClient.has_collection(fake_self, "missing") is False


def test_clone_collection_vectors_chroma_handles_ambiguous_array_values(monkeypatch):
    files = [_file("file-1"), _file("file-2")]

    source_payload = {
        "ids": AmbiguousArray(["chunk-1", "chunk-2"]),
        "embeddings": AmbiguousArray([[0.1, 0.2], [0.3, 0.4]]),
        "documents": AmbiguousArray(["doc1", "doc2"]),
        "metadatas": AmbiguousArray([
            {"file_id": "file-1"},
            {"file_id": "file-2"},
        ]),
    }

    fake_client = FakeVectorClient(
        client=FakeChromaClient(source_payload),
        existing_collections={"collection-source"},
    )

    process_calls = []

    async def fake_process_files_batch(request, form_data, user):
        process_calls.append(form_data)
        return FakeBatchResult([])

    monkeypatch.setattr(knowledge_router, "VECTOR_DB", "chroma")
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", fake_client)
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda knowledge_id, meta: f"collection-{knowledge_id}",
    )
    monkeypatch.setattr(knowledge_router, "process_files_batch", fake_process_files_batch)

    result = asyncio.run(
        knowledge_router._clone_collection_vectors(
            request=SimpleNamespace(),
            source_knowledge=SimpleNamespace(id="source", meta={}),
            target_knowledge=SimpleNamespace(id="target", meta={}),
            files=files,
            user=SimpleNamespace(id="u1"),
        )
    )

    assert result["successful_file_ids"] == ["file-1", "file-2"]
    assert result["warnings"]["strategy"] == "direct_copy"
    assert result["warnings"]["copied_count"] == 2
    assert result["warnings"]["total_items"] == 2
    assert process_calls == []


def test_clone_collection_vectors_chroma_uses_paged_get(monkeypatch):
    files = [_file("file-1"), _file("file-2"), _file("file-3")]

    dataset = {
        "ids": ["chunk-1", "chunk-2", "chunk-3"],
        "embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        "documents": ["doc1", "doc2", "doc3"],
        "metadatas": [
            {"file_id": "file-1"},
            {"file_id": "file-2"},
            {"file_id": "file-3"},
        ],
    }

    def paged_payload(include=None, limit=None, offset=None, ids=None, count_only=False):
        if count_only:
            return len(dataset["ids"])

        start = offset or 0
        end = start + (limit or len(dataset["ids"]))
        return {
            "ids": dataset["ids"][start:end],
            "embeddings": dataset["embeddings"][start:end],
            "documents": dataset["documents"][start:end],
            "metadatas": dataset["metadatas"][start:end],
        }

    fake_chroma = FakeChromaClient(source_payload=paged_payload)
    fake_client = FakeVectorClient(
        client=fake_chroma,
        existing_collections={"collection-source"},
    )

    process_calls = []

    async def fake_process_files_batch(request, form_data, user):
        process_calls.append(form_data)
        return FakeBatchResult([])

    monkeypatch.setattr(knowledge_router, "VECTOR_DB", "chroma")
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", fake_client)
    monkeypatch.setattr(knowledge_router, "CHROMA_CLONE_PAGE_SIZE", 2)
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda knowledge_id, meta: f"collection-{knowledge_id}",
    )
    monkeypatch.setattr(knowledge_router, "process_files_batch", fake_process_files_batch)

    result = asyncio.run(
        knowledge_router._clone_collection_vectors(
            request=SimpleNamespace(),
            source_knowledge=SimpleNamespace(id="source", meta={}),
            target_knowledge=SimpleNamespace(id="target", meta={}),
            files=files,
            user=SimpleNamespace(id="u1"),
        )
    )

    assert result["warnings"]["strategy"] == "direct_copy"
    assert result["warnings"]["copied_count"] == 3
    assert result["warnings"]["total_items"] == 3
    assert result["warnings"]["pages_scanned"] == 2
    assert process_calls == []
    page_calls = [call for call in fake_chroma.source_collection.calls if call["limit"] is not None]
    assert [call["offset"] for call in page_calls] == [0, 2]


def test_clone_collection_vectors_chroma_reports_failed_offset(monkeypatch):
    files = [_file("file-1"), _file("file-2")]

    dataset = {
        "ids": ["chunk-1", "chunk-2"],
        "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        "documents": ["doc1", "doc2"],
        "metadatas": [{"file_id": "file-1"}, {"file_id": "file-2"}],
    }

    def paged_payload(include=None, limit=None, offset=None, ids=None, count_only=False):
        if count_only:
            return len(dataset["ids"])
        if offset == 1:
            raise RuntimeError("synthetic page failure")

        start = offset or 0
        end = start + (limit or len(dataset["ids"]))
        return {
            "ids": dataset["ids"][start:end],
            "embeddings": dataset["embeddings"][start:end],
            "documents": dataset["documents"][start:end],
            "metadatas": dataset["metadatas"][start:end],
        }

    fake_chroma = FakeChromaClient(source_payload=paged_payload)
    fake_client = FakeVectorClient(
        client=fake_chroma,
        existing_collections={"collection-source", "collection-target"},
    )

    async def fake_process_files_batch(request, form_data, user):
        return FakeBatchResult(["file-1", "file-2"])

    monkeypatch.setattr(knowledge_router, "VECTOR_DB", "chroma")
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", fake_client)
    monkeypatch.setattr(knowledge_router, "CHROMA_CLONE_PAGE_SIZE", 1)
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda knowledge_id, meta: f"collection-{knowledge_id}",
    )
    monkeypatch.setattr(knowledge_router, "process_files_batch", fake_process_files_batch)

    result = asyncio.run(
        knowledge_router._clone_collection_vectors(
            request=SimpleNamespace(),
            source_knowledge=SimpleNamespace(id="source", meta={}),
            target_knowledge=SimpleNamespace(id="target", meta={}),
            files=files,
            user=SimpleNamespace(id="u1"),
        )
    )

    assert result["warnings"]["strategy"] == "full_reembed_fallback"
    assert "offset 1" in result["warnings"]["reason"]
    assert result["warnings"]["failed_stage"] == "get_page"
    assert result["warnings"]["failed_offset"] == 1


def test_clone_collection_vectors_chroma_file_id_retry_success(monkeypatch):
    files = [_file("file-1"), _file("file-2")]

    dataset = {
        "file-1": {
            "ids": ["chunk-1"],
            "embeddings": [[0.1, 0.2]],
            "documents": ["doc1"],
            "metadatas": [{"file_id": "file-1"}],
        },
        "file-2": {
            "ids": ["chunk-2"],
            "embeddings": [[0.3, 0.4]],
            "documents": ["doc2"],
            "metadatas": [{"file_id": "file-2"}],
        },
    }

    def paged_payload(include=None, limit=None, offset=None, ids=None, where=None, count_only=False):
        if count_only:
            return 2
        if offset is not None:
            raise RuntimeError("synthetic page failure")
        if where and "file_id" in where:
            return dataset.get(where["file_id"], {"ids": [], "embeddings": [], "documents": [], "metadatas": []})
        return {"ids": [], "embeddings": [], "documents": [], "metadatas": []}

    fake_chroma = FakeChromaClient(source_payload=paged_payload)
    fake_client = FakeVectorClient(
        client=fake_chroma,
        existing_collections={"collection-source"},
    )

    async def fake_process_files_batch(request, form_data, user):
        return FakeBatchResult([])

    monkeypatch.setattr(knowledge_router, "VECTOR_DB", "chroma")
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", fake_client)
    monkeypatch.setattr(knowledge_router, "CHROMA_CLONE_PAGE_SIZE", 1)
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda knowledge_id, meta: f"collection-{knowledge_id}",
    )
    monkeypatch.setattr(knowledge_router, "process_files_batch", fake_process_files_batch)

    result = asyncio.run(
        knowledge_router._clone_collection_vectors(
            request=SimpleNamespace(),
            source_knowledge=SimpleNamespace(id="source", meta={}),
            target_knowledge=SimpleNamespace(id="target", meta={}),
            files=files,
            user=SimpleNamespace(id="u1"),
        )
    )

    assert result["warnings"]["strategy"] == "direct_copy"
    assert result["warnings"]["retry_mode"] == "file_id"
    assert result["warnings"]["file_id_retry_attempts"] == 5
    assert result["warnings"]["file_id_failed_count"] == 0


def test_clone_collection_vectors_chroma_file_id_retry_partial(monkeypatch):
    files = [_file("file-1"), _file("file-2")]

    dataset = {
        "file-1": {
            "ids": ["chunk-1"],
            "embeddings": [[0.1, 0.2]],
            "documents": ["doc1"],
            "metadatas": [{"file_id": "file-1"}],
        },
        "file-2": {"ids": [], "embeddings": [], "documents": [], "metadatas": []},
    }

    def paged_payload(include=None, limit=None, offset=None, ids=None, where=None, count_only=False):
        if count_only:
            return 2
        if offset is not None:
            raise RuntimeError("synthetic page failure")
        if where and "file_id" in where:
            return dataset.get(where["file_id"], {"ids": [], "embeddings": [], "documents": [], "metadatas": []})
        return {"ids": [], "embeddings": [], "documents": [], "metadatas": []}

    fake_chroma = FakeChromaClient(source_payload=paged_payload)
    fake_client = FakeVectorClient(
        client=fake_chroma,
        existing_collections={"collection-source", "collection-target"},
    )

    async def fake_process_files_batch(request, form_data, user):
        return FakeBatchResult([file.id for file in form_data.files])

    monkeypatch.setattr(knowledge_router, "VECTOR_DB", "chroma")
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", fake_client)
    monkeypatch.setattr(knowledge_router, "CHROMA_CLONE_PAGE_SIZE", 1)
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda knowledge_id, meta: f"collection-{knowledge_id}",
    )
    monkeypatch.setattr(knowledge_router, "process_files_batch", fake_process_files_batch)

    result = asyncio.run(
        knowledge_router._clone_collection_vectors(
            request=SimpleNamespace(),
            source_knowledge=SimpleNamespace(id="source", meta={}),
            target_knowledge=SimpleNamespace(id="target", meta={}),
            files=files,
            user=SimpleNamespace(id="u1"),
        )
    )

    assert result["warnings"]["strategy"] == "partial_copy_with_reembed"
    assert result["warnings"]["retry_mode"] == "file_id"
    assert result["warnings"]["file_id_retry_attempts"] == 5
    assert result["warnings"]["file_id_failed_count"] == 0
    assert result["warnings"]["reembedded_file_ids"] == ["file-2"]


def test_clone_collection_vectors_chroma_file_id_retry_failed(monkeypatch):
    files = [_file("file-1"), _file("file-2")]

    def paged_payload(include=None, limit=None, offset=None, ids=None, where=None, count_only=False):
        if count_only:
            return 2
        if offset is not None:
            raise RuntimeError("synthetic page failure")
        if where and "file_id" in where:
            raise RuntimeError("synthetic file_id failure")
        return {"ids": [], "embeddings": [], "documents": [], "metadatas": []}

    fake_chroma = FakeChromaClient(source_payload=paged_payload)
    fake_client = FakeVectorClient(
        client=fake_chroma,
        existing_collections={"collection-source", "collection-target"},
    )

    async def fake_process_files_batch(request, form_data, user):
        return FakeBatchResult([file.id for file in form_data.files])

    monkeypatch.setattr(knowledge_router, "VECTOR_DB", "chroma")
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", fake_client)
    monkeypatch.setattr(knowledge_router, "CHROMA_CLONE_PAGE_SIZE", 1)
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda knowledge_id, meta: f"collection-{knowledge_id}",
    )
    monkeypatch.setattr(knowledge_router, "process_files_batch", fake_process_files_batch)

    result = asyncio.run(
        knowledge_router._clone_collection_vectors(
            request=SimpleNamespace(),
            source_knowledge=SimpleNamespace(id="source", meta={}),
            target_knowledge=SimpleNamespace(id="target", meta={}),
            files=files,
            user=SimpleNamespace(id="u1"),
        )
    )

    assert result["warnings"]["strategy"] == "full_reembed_fallback"
    assert result["warnings"]["retry_mode"] == "file_id"
    assert result["warnings"]["file_id_retry_attempts"] == 5
    assert result["warnings"]["file_id_failed_count"] == 2
