from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from open_webui.routers import files as files_router
from open_webui.routers import retrieval as retrieval_router


def _mock_file(file_id: str = "file-1", content: str = "stored-content"):
    return SimpleNamespace(
        id=file_id,
        user_id="user-1",
        filename="doc.txt",
        meta={},
        data={"content": content},
    )


@pytest.mark.asyncio
async def test_update_file_data_content_uses_threadpool(monkeypatch):
    file_obj = _mock_file()
    user = SimpleNamespace(id="user-1", role="user")
    request = SimpleNamespace()
    calls = {}

    monkeypatch.setattr(files_router.Files, "get_file_by_id", lambda id: file_obj)

    def fake_process_file(req, form_data, passed_user):
        calls["process_file"] = (req, form_data, passed_user)

    async def fake_run_in_threadpool(func, *args):
        calls["threadpool"] = {"func": func, "args": args}
        return func(*args)

    monkeypatch.setattr(files_router, "process_file", fake_process_file)
    monkeypatch.setattr(files_router, "run_in_threadpool", fake_run_in_threadpool)

    result = await files_router.update_file_data_content_by_id(
        request=request,
        id=file_obj.id,
        form_data=files_router.ContentForm(content="updated content"),
        user=user,
    )

    assert result == {"content": "stored-content"}
    assert calls["threadpool"]["func"] == fake_process_file
    assert calls["threadpool"]["args"][2] == user
    assert calls["process_file"][1].content == "updated content"


@pytest.mark.asyncio
async def test_update_file_data_content_returns_400_on_processing_error(monkeypatch):
    file_obj = _mock_file()
    user = SimpleNamespace(id="user-1", role="user")
    request = SimpleNamespace()

    monkeypatch.setattr(files_router.Files, "get_file_by_id", lambda id: file_obj)

    async def fake_run_in_threadpool(func, *args):
        raise RuntimeError("boom")

    monkeypatch.setattr(files_router, "run_in_threadpool", fake_run_in_threadpool)

    with pytest.raises(HTTPException) as exc_info:
        await files_router.update_file_data_content_by_id(
            request=request,
            id=file_obj.id,
            form_data=files_router.ContentForm(content="updated content"),
            user=user,
        )

    assert exc_info.value.status_code == 400
    assert "boom" in exc_info.value.detail


def test_process_file_skips_delete_when_collection_missing(monkeypatch):
    class FakeVectorClient:
        def __init__(self):
            self.has_collection_calls = []
            self.delete_collection_calls = []

        def has_collection(self, collection_name):
            self.has_collection_calls.append(collection_name)
            return False

        def delete_collection(self, collection_name):
            self.delete_collection_calls.append(collection_name)

    fake_vector_client = FakeVectorClient()
    file_obj = _mock_file(file_id="f-123")
    user = SimpleNamespace(id="admin-1", role="admin")
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                config=SimpleNamespace(
                    CONVERSATION_FILE_UPLOAD_EMBEDDING=False,
                    BYPASS_EMBEDDING_AND_RETRIEVAL=True,
                )
            )
        )
    )

    monkeypatch.setattr(retrieval_router, "VECTOR_DB_CLIENT", fake_vector_client)
    monkeypatch.setattr(
        retrieval_router.Files, "get_file_by_id", lambda file_id: file_obj
    )
    monkeypatch.setattr(
        retrieval_router,
        "is_conversation_file_upload_embedding_enabled",
        lambda user, global_enabled: False,
    )
    monkeypatch.setattr(
        retrieval_router,
        "get_collection_effective_config",
        lambda request, collection_name: {"effective": {}},
    )
    monkeypatch.setattr(
        retrieval_router.Files, "update_file_data_by_id", lambda file_id, data: None
    )
    monkeypatch.setattr(
        retrieval_router.Files, "update_file_hash_by_id", lambda file_id, hash_value: None
    )

    result = retrieval_router.process_file(
        request,
        retrieval_router.ProcessFileForm(file_id=file_obj.id, content="new content"),
        user=user,
    )

    assert result["status"] is True
    assert fake_vector_client.has_collection_calls == [f"file-{file_obj.id}"]
    assert fake_vector_client.delete_collection_calls == []
