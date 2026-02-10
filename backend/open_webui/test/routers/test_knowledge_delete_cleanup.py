import asyncio
from types import SimpleNamespace

import pytest

from open_webui.routers import knowledge as knowledge_router


class DummyKnowledge:
    def __init__(self, knowledge_id: str, name: str = "k"):
        self.id = knowledge_id
        self.name = name
        self.user_id = "u1"
        self.meta = {}
        self.access_control = None

    def model_dump(self):
        return {
            "id": self.id,
            "name": self.name,
            "user_id": self.user_id,
            "description": "d",
            "meta": self.meta,
            "access_control": self.access_control,
            "created_at": 0,
            "updated_at": 0,
        }


class DummyVectorClient:
    def __init__(self):
        self.deleted = []
        self.deleted_collections = []
        self.existing = set()

    def delete(self, collection_name, filter):
        self.deleted.append((collection_name, filter))

    def has_collection(self, collection_name):
        return collection_name in self.existing

    def delete_collection(self, collection_name):
        self.deleted_collections.append(collection_name)


def _file(file_id: str):
    return SimpleNamespace(id=file_id, hash="h1", path="/tmp/file.txt")


def _user():
    return SimpleNamespace(id="u1", role="user")


def test_hard_delete_file_everywhere_success(monkeypatch):
    vector = DummyVectorClient()
    vector.existing.add("file-file-1")

    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", vector)
    monkeypatch.setattr(knowledge_router.Files, "get_file_by_id", lambda _id: _file(_id))
    monkeypatch.setattr(knowledge_router.Files, "delete_file_by_id", lambda _id: True)
    monkeypatch.setattr(
        knowledge_router.Knowledges,
        "get_knowledges_by_file_id",
        lambda _id: [DummyKnowledge("k1"), DummyKnowledge("k2")],
    )
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda knowledge_id, _meta: f"col-{knowledge_id}",
    )

    storage_deleted = []
    bm25_dirty = []
    bm25_invalid = []

    monkeypatch.setattr(
        knowledge_router.Storage,
        "delete_file",
        lambda file_path: storage_deleted.append(file_path),
    )
    monkeypatch.setattr(
        knowledge_router,
        "mark_bm25_collections_dirty",
        lambda names: bm25_dirty.extend(names),
    )
    monkeypatch.setattr(
        knowledge_router,
        "invalidate_bm25_collections",
        lambda names: bm25_invalid.extend(names),
    )

    warnings = knowledge_router._hard_delete_file_everywhere("file-1")

    assert warnings == []
    assert ("col-k1", {"file_id": "file-1"}) in vector.deleted
    assert ("col-k1", {"hash": "h1"}) in vector.deleted
    assert ("col-k2", {"file_id": "file-1"}) in vector.deleted
    assert ("col-k2", {"hash": "h1"}) in vector.deleted
    assert vector.deleted_collections == ["file-file-1"]
    assert storage_deleted == ["/tmp/file.txt"]
    assert set(bm25_dirty) == {"col-k1", "col-k2"}
    assert set(bm25_invalid) == {"col-k1", "col-k2"}


def test_hard_delete_file_everywhere_collects_warnings(monkeypatch):
    vector = DummyVectorClient()

    def failing_delete(collection_name, filter):
        raise RuntimeError(f"fail {collection_name} {filter}")

    vector.delete = failing_delete

    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", vector)
    monkeypatch.setattr(knowledge_router.Files, "get_file_by_id", lambda _id: _file(_id))
    monkeypatch.setattr(knowledge_router.Files, "delete_file_by_id", lambda _id: False)
    monkeypatch.setattr(
        knowledge_router.Knowledges,
        "get_knowledges_by_file_id",
        lambda _id: [DummyKnowledge("k1")],
    )
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda knowledge_id, _meta: f"col-{knowledge_id}",
    )
    monkeypatch.setattr(
        knowledge_router.Storage,
        "delete_file",
        lambda _path: (_ for _ in ()).throw(RuntimeError("storage fail")),
    )
    monkeypatch.setattr(knowledge_router, "mark_bm25_collections_dirty", lambda _names: None)
    monkeypatch.setattr(knowledge_router, "invalidate_bm25_collections", lambda _names: None)

    warnings = knowledge_router._hard_delete_file_everywhere("file-1")

    stages = [warning["stage"] for warning in warnings]
    assert "vector.delete_by_file_id" in stages
    assert "vector.delete_by_hash" in stages
    assert "storage.delete_file" in stages
    assert "db.delete_file" in stages


def test_remove_file_shared_reference_skips_entity_delete(monkeypatch):
    knowledge = DummyKnowledge("k1")

    monkeypatch.setattr(knowledge_router.Knowledges, "get_knowledge_by_id", lambda _id: knowledge)
    monkeypatch.setattr(knowledge_router.Files, "get_file_by_id", lambda _id: _file(_id))
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda _id, _meta: "col-k1",
    )

    detached = []
    deleted = []
    vector = DummyVectorClient()
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", vector)
    monkeypatch.setattr(
        knowledge_router.Knowledges,
        "remove_file_from_knowledge_by_id",
        lambda knowledge_id, file_id: detached.append((knowledge_id, file_id)) or True,
    )
    monkeypatch.setattr(knowledge_router.Files, "delete_file_by_id", lambda _id: deleted.append(_id) or True)
    monkeypatch.setattr(knowledge_router, "mark_bm25_collections_dirty", lambda _names: None)
    monkeypatch.setattr(knowledge_router, "invalidate_bm25_collections", lambda _names: None)
    monkeypatch.setattr(
        knowledge_router,
        "_get_other_collection_references",
        lambda _file_id, _knowledge_id: [DummyKnowledge("k2", "other")],
    )
    monkeypatch.setattr(knowledge_router.Knowledges, "get_file_metadatas_by_id", lambda _id: [])

    res = knowledge_router.remove_file_from_knowledge_by_id(
        "k1",
        knowledge_router.KnowledgeFileIdForm(file_id="file-1"),
        force=False,
        user=_user(),
    )

    assert detached == [("k1", "file-1")]
    assert deleted == []
    assert ("col-k1", {"file_id": "file-1"}) in vector.deleted
    assert res.warnings is not None
    assert res.warnings["skipped_file_deletions_count"] == 1
    assert res.warnings["skipped_file_deletions_sample"][0]["file_id"] == "file-1"


def test_remove_file_force_delete(monkeypatch):
    knowledge = DummyKnowledge("k1")

    monkeypatch.setattr(knowledge_router.Knowledges, "get_knowledge_by_id", lambda _id: knowledge)
    monkeypatch.setattr(knowledge_router.Files, "get_file_by_id", lambda _id: _file(_id))
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda _id, _meta: "col-k1",
    )

    detached = []
    deleted = []
    vector = DummyVectorClient()
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", vector)
    monkeypatch.setattr(
        knowledge_router.Knowledges,
        "remove_file_from_knowledge_by_id",
        lambda knowledge_id, file_id: detached.append((knowledge_id, file_id)) or True,
    )
    monkeypatch.setattr(knowledge_router.Files, "delete_file_by_id", lambda _id: deleted.append(_id) or True)
    monkeypatch.setattr(knowledge_router, "mark_bm25_collections_dirty", lambda _names: None)
    monkeypatch.setattr(knowledge_router, "invalidate_bm25_collections", lambda _names: None)
    monkeypatch.setattr(
        knowledge_router,
        "_get_other_collection_references",
        lambda _file_id, _knowledge_id: [DummyKnowledge("k2", "other")],
    )
    monkeypatch.setattr(knowledge_router.Knowledges, "get_file_metadatas_by_id", lambda _id: [])

    res = knowledge_router.remove_file_from_knowledge_by_id(
        "k1",
        knowledge_router.KnowledgeFileIdForm(file_id="file-1"),
        force=True,
        user=_user(),
    )

    assert detached == [("k1", "file-1")]
    assert deleted == []
    assert ("col-k1", {"file_id": "file-1"}) in vector.deleted
    assert res.warnings is not None
    assert res.warnings["skipped_file_deletions_count"] == 1


def test_remove_file_non_shared_deletes_entity(monkeypatch):
    knowledge = DummyKnowledge("k1")

    monkeypatch.setattr(knowledge_router.Knowledges, "get_knowledge_by_id", lambda _id: knowledge)
    monkeypatch.setattr(knowledge_router.Files, "get_file_by_id", lambda _id: _file(_id))
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda _id, _meta: "col-k1",
    )
    monkeypatch.setattr(
        knowledge_router,
        "_get_other_collection_references",
        lambda _file_id, _knowledge_id: [],
    )
    monkeypatch.setattr(knowledge_router.Knowledges, "get_file_metadatas_by_id", lambda _id: [])

    detached = []
    deleted = []
    storage_deleted = []
    vector = DummyVectorClient()
    vector.existing.add("file-file-1")
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", vector)
    monkeypatch.setattr(
        knowledge_router.Knowledges,
        "remove_file_from_knowledge_by_id",
        lambda knowledge_id, file_id: detached.append((knowledge_id, file_id)) or True,
    )
    monkeypatch.setattr(knowledge_router.Files, "delete_file_by_id", lambda _id: deleted.append(_id) or True)
    monkeypatch.setattr(
        knowledge_router.Storage,
        "delete_file",
        lambda file_path: storage_deleted.append(file_path),
    )
    monkeypatch.setattr(knowledge_router, "mark_bm25_collections_dirty", lambda _names: None)
    monkeypatch.setattr(knowledge_router, "invalidate_bm25_collections", lambda _names: None)

    res = knowledge_router.remove_file_from_knowledge_by_id(
        "k1",
        knowledge_router.KnowledgeFileIdForm(file_id="file-1"),
        force=False,
        user=_user(),
    )

    assert detached == [("k1", "file-1")]
    assert deleted == ["file-1"]
    assert storage_deleted == ["/tmp/file.txt"]
    assert vector.deleted_collections == ["file-file-1"]
    assert res.warnings is None


def test_delete_collection_shared_files_are_skipped_for_any_force_value(monkeypatch):
    knowledge = DummyKnowledge("k1")

    monkeypatch.setattr(knowledge_router.Knowledges, "get_knowledge_by_id", lambda _id: knowledge)
    monkeypatch.setattr(
        knowledge_router.Knowledges,
        "get_files_by_id",
        lambda _id: [SimpleNamespace(id="file-1")],
    )
    monkeypatch.setattr(knowledge_router.Models, "get_all_models", lambda: [])
    monkeypatch.setattr(knowledge_router.Knowledges, "delete_knowledge_by_id", lambda _id: True)
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda _id, _meta: "col-k1",
    )

    vector = DummyVectorClient()
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", vector)
    monkeypatch.setattr(knowledge_router, "mark_bm25_collections_dirty", lambda _names: None)
    monkeypatch.setattr(knowledge_router, "invalidate_bm25_collections", lambda _names: None)
    deleted_entities = []
    monkeypatch.setattr(
        knowledge_router,
        "_delete_file_entity",
        lambda _file_id: deleted_entities.append(_file_id) or [],
    )

    monkeypatch.setattr(
        knowledge_router,
        "_get_other_collection_references",
        lambda _file_id, _knowledge_id: [DummyKnowledge("k2", "other")],
    )

    res = asyncio.run(knowledge_router.delete_knowledge_by_id("k1", force=False, user=_user()))

    assert res.status is True
    assert deleted_entities == []
    assert res.warnings is not None
    assert res.warnings["skipped_file_deletions_count"] == 1

    force_res = asyncio.run(knowledge_router.delete_knowledge_by_id("k1", force=True, user=_user()))
    assert force_res.status is True
    assert force_res.warnings is not None
    assert force_res.warnings["skipped_file_deletions_count"] == 1


def test_delete_collection_mixed_files(monkeypatch):
    knowledge = DummyKnowledge("k1")

    monkeypatch.setattr(knowledge_router.Knowledges, "get_knowledge_by_id", lambda _id: knowledge)
    monkeypatch.setattr(
        knowledge_router.Knowledges,
        "get_files_by_id",
        lambda _id: [SimpleNamespace(id="file-a"), SimpleNamespace(id="file-b")],
    )
    monkeypatch.setattr(knowledge_router.Models, "get_all_models", lambda: [])
    monkeypatch.setattr(knowledge_router.Knowledges, "delete_knowledge_by_id", lambda _id: True)
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda _id, _meta: "col-k1",
    )

    vector = DummyVectorClient()
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", vector)
    monkeypatch.setattr(knowledge_router, "mark_bm25_collections_dirty", lambda _names: None)
    monkeypatch.setattr(knowledge_router, "invalidate_bm25_collections", lambda _names: None)

    monkeypatch.setattr(
        knowledge_router,
        "_get_other_collection_references",
        lambda file_id, _knowledge_id: [DummyKnowledge("k2", "other")]
        if file_id == "file-a"
        else [],
    )

    deleted_entities = []
    monkeypatch.setattr(
        knowledge_router,
        "_delete_file_entity",
        lambda _file_id: deleted_entities.append(_file_id) or [],
    )

    res = asyncio.run(knowledge_router.delete_knowledge_by_id("k1", force=False, user=_user()))

    assert res.status is True
    assert deleted_entities == ["file-b"]
    assert res.warnings is not None
    assert res.warnings["skipped_file_deletions_count"] == 1
    assert res.warnings["skipped_file_deletions_sample"][0]["file_id"] == "file-a"


def test_detach_shared_file_does_not_touch_other_collections(monkeypatch):
    knowledge = DummyKnowledge("k1")

    monkeypatch.setattr(knowledge_router.Knowledges, "get_knowledge_by_id", lambda _id: knowledge)
    monkeypatch.setattr(knowledge_router.Files, "get_file_by_id", lambda _id: _file(_id))
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda _id, _meta: "col-k1",
    )
    monkeypatch.setattr(
        knowledge_router,
        "_get_other_collection_references",
        lambda _file_id, _knowledge_id: [DummyKnowledge("k2", "other")],
    )
    monkeypatch.setattr(knowledge_router.Knowledges, "get_file_metadatas_by_id", lambda _id: [])
    monkeypatch.setattr(
        knowledge_router.Knowledges,
        "remove_file_from_knowledge_by_id",
        lambda _knowledge_id, _file_id: True,
    )
    monkeypatch.setattr(knowledge_router, "mark_bm25_collections_dirty", lambda _names: None)
    monkeypatch.setattr(knowledge_router, "invalidate_bm25_collections", lambda _names: None)

    vector = DummyVectorClient()
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", vector)

    knowledge_router.remove_file_from_knowledge_by_id(
        "k1",
        knowledge_router.KnowledgeFileIdForm(file_id="file-1"),
        force=False,
        user=_user(),
    )

    touched_collections = {name for name, _filter in vector.deleted}
    assert touched_collections == {"col-k1"}
