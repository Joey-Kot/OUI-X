import asyncio
import json
import os
from types import SimpleNamespace
import zipfile

import pytest

from open_webui.models.files import FileModel
from open_webui.routers import knowledge as knowledge_router


class DummyKnowledge:
    def __init__(self, knowledge_id: str, user_id: str = "u1", name: str = "Knowledge"):
        self.id = knowledge_id
        self.user_id = user_id
        self.name = name
        self.meta = {}
        self.access_control = None


class DummyVectorClient:
    def __init__(self, payload_by_collection=None):
        self.payload_by_collection = payload_by_collection or {}

    def query(self, collection_name, filter, limit=None):
        payload = self.payload_by_collection.get(collection_name)
        if payload is None:
            return None

        return SimpleNamespace(
            ids=[[f"{collection_name}-0"]],
            documents=[payload],
            metadatas=[[{"file_id": filter.get("file_id")}]],
        )


def _file(file_id: str, filename: str, path: str, content_type: str = "text/plain"):
    return FileModel(
        id=file_id,
        user_id="u1",
        hash=None,
        filename=filename,
        path=path,
        data={},
        meta={"name": filename, "content_type": content_type},
        access_control=None,
        created_at=0,
        updated_at=0,
    )


def _read_manifest(zip_path: str):
    with zipfile.ZipFile(zip_path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        names = sorted(archive.namelist())
    return manifest, names


def test_download_all_from_storage(monkeypatch, tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    files = [
        _file("f1", "a.txt", str(first)),
        _file("f2", "b.txt", str(second)),
    ]

    monkeypatch.setattr(knowledge_router.Storage, "get_file", lambda file_path: file_path)
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", DummyVectorClient())

    zip_path, _ = knowledge_router._build_knowledge_download_zip(DummyKnowledge("k1"), files)

    manifest, names = _read_manifest(zip_path)
    os.remove(zip_path)

    assert manifest["summary"] == {
        "total": 2,
        "from_storage": 2,
        "from_vector": 0,
        "failed": 0,
    }
    assert "files/a.txt" in names
    assert "files/b.txt" in names


def test_download_fallback_to_vector_when_storage_missing(monkeypatch):
    file = _file("f1", "missing.pdf", "/tmp/does-not-exist")

    def _raise_missing(_file_path):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(knowledge_router.Storage, "get_file", _raise_missing)
    monkeypatch.setattr(
        knowledge_router,
        "VECTOR_DB_CLIENT",
        DummyVectorClient(payload_by_collection={"file-f1": ["vector-content"]}),
    )

    zip_path, _ = knowledge_router._build_knowledge_download_zip(DummyKnowledge("k1"), [file])

    manifest, names = _read_manifest(zip_path)
    with zipfile.ZipFile(zip_path, "r") as archive:
        fallback_text = archive.read("files/missing.pdf.fallback.txt").decode("utf-8")
    os.remove(zip_path)

    assert fallback_text == "vector-content"
    assert manifest["summary"]["from_vector"] == 1
    assert manifest["items"][0]["source"] == "vector"
    assert "files/missing.pdf.fallback.txt" in names


def test_download_fallback_to_active_collection_when_file_collection_empty(monkeypatch):
    file = _file("f1", "missing.pdf", "/tmp/does-not-exist")

    def _raise_missing(_file_path):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(knowledge_router.Storage, "get_file", _raise_missing)
    monkeypatch.setattr(
        knowledge_router,
        "VECTOR_DB_CLIENT",
        DummyVectorClient(payload_by_collection={"collection-k1": ["active-vector"]}),
    )
    monkeypatch.setattr(
        knowledge_router,
        "get_active_vector_collection_name",
        lambda _knowledge_id, _meta: "collection-k1",
    )

    zip_path, _ = knowledge_router._build_knowledge_download_zip(DummyKnowledge("k1"), [file])

    manifest, _ = _read_manifest(zip_path)
    os.remove(zip_path)

    assert manifest["summary"]["from_vector"] == 1
    assert manifest["items"][0]["source"] == "vector"


def test_download_partial_failure_manifest_records_errors(monkeypatch, tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("ok", encoding="utf-8")

    files = [
        _file("ok", "good.txt", str(good)),
        _file("bad", "bad.txt", "/tmp/missing"),
    ]

    def _resolve(file_path):
        if file_path == "/tmp/missing":
            raise FileNotFoundError("missing")
        return file_path

    monkeypatch.setattr(knowledge_router.Storage, "get_file", _resolve)
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", DummyVectorClient())

    zip_path, _ = knowledge_router._build_knowledge_download_zip(DummyKnowledge("k1"), files)

    manifest, names = _read_manifest(zip_path)
    os.remove(zip_path)

    assert manifest["summary"]["from_storage"] == 1
    assert manifest["summary"]["failed"] == 1
    failed_items = [item for item in manifest["items"] if item["source"] == "fail"]
    assert len(failed_items) == 1
    assert failed_items[0]["error"]
    assert "files/good.txt" in names


def test_download_requires_read_access(monkeypatch):
    knowledge = DummyKnowledge("k1", user_id="owner")
    user = SimpleNamespace(id="reader", role="user")

    monkeypatch.setattr(knowledge_router.Knowledges, "get_knowledge_by_id", lambda _id: knowledge)
    monkeypatch.setattr(knowledge_router, "has_access", lambda *_args, **_kwargs: False)

    with pytest.raises(knowledge_router.HTTPException) as exc_info:
        asyncio.run(knowledge_router.download_knowledge_by_id("k1", user=user))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == knowledge_router.ERROR_MESSAGES.ACCESS_PROHIBITED


def test_zip_filename_strips_uuid_prefix(monkeypatch, tmp_path):
    file_path = tmp_path / "data.bin"
    file_path.write_bytes(b"payload")

    file = FileModel(
        id="f1",
        user_id="u1",
        hash=None,
        filename="123e4567-e89b-12d3-a456-426614174000_report.pdf",
        path=str(file_path),
        data={},
        meta={"content_type": "application/pdf"},
        access_control=None,
        created_at=0,
        updated_at=0,
    )

    monkeypatch.setattr(knowledge_router.Storage, "get_file", lambda file_path: file_path)
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", DummyVectorClient())

    zip_path, _ = knowledge_router._build_knowledge_download_zip(DummyKnowledge("k1"), [file])

    _, names = _read_manifest(zip_path)
    os.remove(zip_path)

    assert "files/report.pdf" in names
    assert "files/123e4567-e89b-12d3-a456-426614174000_report.pdf" not in names


def test_zip_filename_deduplication(monkeypatch, tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")

    files = [
        _file("f1", "same.txt", str(first)),
        _file("f2", "same.txt", str(second)),
    ]

    monkeypatch.setattr(knowledge_router.Storage, "get_file", lambda file_path: file_path)
    monkeypatch.setattr(knowledge_router, "VECTOR_DB_CLIENT", DummyVectorClient())

    zip_path, _ = knowledge_router._build_knowledge_download_zip(DummyKnowledge("k1"), files)

    _, names = _read_manifest(zip_path)
    os.remove(zip_path)

    assert "files/same.txt" in names
    assert "files/same (1).txt" in names
