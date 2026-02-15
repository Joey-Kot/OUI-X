from typing import List, Optional
import copy
import json
import os
from pathlib import Path
import re
import tempfile
import time
import uuid
import zipfile
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
import logging
from starlette.background import BackgroundTask
from urllib.parse import quote

from open_webui.models.groups import Groups
from open_webui.models.knowledge import (
    KnowledgeFileListResponse,
    Knowledges,
    KnowledgeForm,
    KnowledgeResponse,
    KnowledgeUserResponse,
)
from open_webui.models.files import Files, FileModel, FileMetadataResponse
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
from open_webui.routers.retrieval import (
    process_file,
    ProcessFileForm,
    process_files_batch,
    BatchProcessFilesForm,
)
from open_webui.storage.provider import Storage

from open_webui.constants import ERROR_MESSAGES
from open_webui.utils.auth import get_verified_user
from open_webui.utils.access_control import has_access, has_permission


from open_webui.config import (
    BYPASS_ADMIN_ACCESS_CONTROL,
    VECTOR_DB,
    ENABLE_QDRANT_MULTITENANCY_MODE,
)
from open_webui.models.models import Models, ModelForm
from open_webui.models.chats import Chats
from open_webui.retrieval.utils import (
    invalidate_bm25_collections,
    mark_bm25_collections_dirty,
)
from open_webui.utils.knowledge import (
    get_active_vector_collection_name,
    resolve_collection_rag_config,
    sanitize_rag_overrides,
    upsert_collection_rag_meta,
    upsert_reindex_status_meta,
)


log = logging.getLogger(__name__)

router = APIRouter()

CHROMA_CLONE_PAGE_SIZE = 200
FILE_ID_RETRY_ATTEMPTS = 5
FILE_ID_RETRY_BACKOFF_BASE = 0.2
FILE_ID_RETRY_BACKOFF_CAP = 2.0
UUID_FILENAME_PREFIX_PATTERN = re.compile(r"^[0-9a-fA-F-]{36}_(.+)$")


SUPPORTED_EMBEDDING_ENGINES = {"openai", "azure_openai"}
SUPPORTED_TEXT_SPLITTERS = {"", "character", "token", "token_voyage"}
SUPPORTED_RERANKING_ENGINES = {"external", "voyage"}


def _safe_display_filename(file: FileModel) -> str:
    meta_name = file.meta.get("name") if isinstance(file.meta, dict) else None
    candidate = meta_name or file.filename or file.id
    candidate = os.path.basename(candidate) if candidate else file.id
    candidate = candidate.strip()

    match = UUID_FILENAME_PREFIX_PATTERN.match(candidate)
    if match:
        candidate = match.group(1)

    candidate = candidate.replace("/", "_").replace("\\", "_").strip(" .")
    return candidate or file.id


def _dedupe_zip_entry_name(entry_name: str, used_entries: set[str]) -> str:
    if entry_name not in used_entries:
        used_entries.add(entry_name)
        return entry_name

    directory, filename = os.path.split(entry_name)
    stem, suffix = os.path.splitext(filename)

    counter = 1
    while True:
        candidate_name = f"{stem} ({counter}){suffix}"
        candidate = os.path.join(directory, candidate_name)
        if candidate not in used_entries:
            used_entries.add(candidate)
            return candidate
        counter += 1


def _resolve_storage_file_bytes(file: FileModel) -> bytes:
    if not file.path:
        raise FileNotFoundError("file path is empty")

    local_path = Storage.get_file(file.path)
    resolved_path = Path(local_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"resolved path does not exist: {resolved_path}")

    return resolved_path.read_bytes()


def _extract_vector_documents(result) -> list[str]:
    if not result:
        return []

    documents = getattr(result, "documents", None)
    if not documents:
        return []

    first_batch = documents[0] if len(documents) > 0 else []
    if not first_batch:
        return []

    return [doc for doc in first_batch if isinstance(doc, str) and doc]


def _resolve_vector_fallback_text(file: FileModel, knowledge) -> str | None:
    collections_to_query = [f"file-{file.id}"]

    active_collection_name = get_active_vector_collection_name(knowledge.id, knowledge.meta)
    if active_collection_name and active_collection_name not in collections_to_query:
        collections_to_query.append(active_collection_name)

    for collection_name in collections_to_query:
        try:
            result = VECTOR_DB_CLIENT.query(
                collection_name=collection_name,
                filter={"file_id": file.id},
            )
        except Exception:
            continue

        documents = _extract_vector_documents(result)
        if documents:
            return "\n\n".join(documents)

    return None


def _build_knowledge_archive_name(knowledge) -> str:
    candidate = (knowledge.name or "").strip()
    if not candidate:
        candidate = f"knowledge-{knowledge.id}"

    candidate = re.sub(r"[\\/:*?\"<>|]+", "_", candidate)
    candidate = candidate.strip(" .")
    if not candidate:
        candidate = f"knowledge-{knowledge.id}"

    return f"{candidate}.zip"


def _remove_temp_file(file_path: str):
    try:
        os.remove(file_path)
    except Exception:
        pass


def _build_knowledge_download_zip(knowledge, files: list[FileModel]) -> tuple[str, dict]:
    zip_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    zip_file.close()

    manifest = {
        "knowledge": {
            "id": knowledge.id,
            "name": knowledge.name,
            "exported_at": int(time.time()),
        },
        "summary": {
            "total": len(files),
            "from_storage": 0,
            "from_vector": 0,
            "failed": 0,
        },
        "items": [],
    }

    used_entries: set[str] = set()
    try:
        with zipfile.ZipFile(zip_file.name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in files:
                display_name = _safe_display_filename(file)
                mime_type = (
                    file.meta.get("content_type")
                    if isinstance(file.meta, dict)
                    else None
                )

                manifest_item = {
                    "file_id": file.id,
                    "original_name": display_name,
                    "zip_entry": None,
                    "source": "fail",
                    "mime_type": mime_type,
                    "size_bytes": 0,
                    "storage_path": file.path,
                    "error": None,
                }

                try:
                    storage_bytes = _resolve_storage_file_bytes(file)
                    zip_entry = _dedupe_zip_entry_name(
                        os.path.join("files", display_name),
                        used_entries,
                    )
                    archive.writestr(zip_entry, storage_bytes)

                    manifest_item["zip_entry"] = zip_entry
                    manifest_item["source"] = "storage"
                    manifest_item["size_bytes"] = len(storage_bytes)
                    manifest["summary"]["from_storage"] += 1
                    manifest["items"].append(manifest_item)
                    continue
                except Exception as storage_error:
                    storage_error_message = str(storage_error)

                fallback_text = _resolve_vector_fallback_text(file, knowledge)
                if fallback_text is not None and fallback_text.strip() != "":
                    fallback_entry = _dedupe_zip_entry_name(
                        os.path.join("files", f"{display_name}.fallback.txt"),
                        used_entries,
                    )
                    fallback_bytes = fallback_text.encode("utf-8")
                    archive.writestr(fallback_entry, fallback_bytes)

                    manifest_item["zip_entry"] = fallback_entry
                    manifest_item["source"] = "vector"
                    manifest_item["size_bytes"] = len(fallback_bytes)
                    manifest["summary"]["from_vector"] += 1
                else:
                    manifest_item["source"] = "fail"
                    manifest_item["error"] = storage_error_message
                    manifest["summary"]["failed"] += 1

                manifest["items"].append(manifest_item)

            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2),
            )

    except Exception:
        _remove_temp_file(zip_file.name)
        raise

    return zip_file.name, manifest

############################
# getKnowledgeBases
############################

PAGE_ITEM_COUNT = 30


class KnowledgeAccessResponse(KnowledgeUserResponse):
    write_access: Optional[bool] = False


class KnowledgeAccessListResponse(BaseModel):
    items: list[KnowledgeAccessResponse]
    total: int


class CollectionRagConfigUpdateForm(BaseModel):
    mode: str = "default"
    overrides: Optional[dict] = None


class DeleteKnowledgeResponse(BaseModel):
    status: bool
    warnings: Optional[dict] = None


def _ensure_write_access(knowledge, user):
    if (
        knowledge.user_id != user.id
        and not has_access(user.id, "write", knowledge.access_control)
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )


def _validate_collection_rag_overrides(overrides: dict):
    if "RAG_EMBEDDING_ENGINE" in overrides and overrides["RAG_EMBEDDING_ENGINE"] not in SUPPORTED_EMBEDDING_ENGINES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid embedding engine",
        )

    if "TEXT_SPLITTER" in overrides and overrides["TEXT_SPLITTER"] not in SUPPORTED_TEXT_SPLITTERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid text splitter",
        )

    if "RAG_RERANKING_ENGINE" in overrides and overrides["RAG_RERANKING_ENGINE"] not in SUPPORTED_RERANKING_ENGINES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid reranking engine",
        )

    for int_key in [
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "RAG_EMBEDDING_BATCH_SIZE",
        "TOP_K",
        "TOP_K_RERANKER",
    ]:
        if int_key in overrides and overrides[int_key] < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{int_key} must be >= 0",
            )

    if "RELEVANCE_THRESHOLD" in overrides and not (
        0.0 <= overrides["RELEVANCE_THRESHOLD"] <= 1.0
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="RELEVANCE_THRESHOLD must be between 0.0 and 1.0",
        )

    if "BM25_WEIGHT" in overrides and not (0.0 <= overrides["BM25_WEIGHT"] <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="BM25_WEIGHT must be between 0.0 and 1.0",
        )

    if "RETRIEVAL_CHUNK_EXPANSION" in overrides and not (
        0 <= overrides["RETRIEVAL_CHUNK_EXPANSION"] <= 100
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="RETRIEVAL_CHUNK_EXPANSION must be between 0 and 100",
        )

    if "RAG_TEMPLATE" in overrides and not isinstance(overrides["RAG_TEMPLATE"], str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="RAG_TEMPLATE must be a string",
        )


def _ensure_knowledge_create_permission(request: Request, user):
    if user.role != "admin" and not has_permission(
        user.id, "workspace.knowledge", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )


def _build_cloned_knowledge_meta(meta: Optional[dict]) -> dict:
    # Clone functional metadata but never carry source vector-state pointers.
    next_meta = copy.deepcopy(meta or {})
    next_meta.pop("collection_vector", None)
    return next_meta


def _extract_qdrant_vector(point_vector):
    if isinstance(point_vector, dict):
        for value in point_vector.values():
            if value is not None:
                return value
        return None
    return point_vector


def _chunk_vector_items(items: list, chunk_size: int = 256):
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _upsert_chroma_records(
    target_collection_name: str,
    ids: list,
    embeddings: list,
    documents: list,
    metadatas: list,
):
    items = []
    for idx, item_id in enumerate(ids):
        items.append(
            {
                "id": item_id,
                "text": documents[idx] if idx < len(documents) else "",
                "vector": embeddings[idx],
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
            }
        )

    if items:
        VECTOR_DB_CLIENT.upsert(collection_name=target_collection_name, items=items)


def _normalize_chroma_sequence(value) -> list:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            normalized = tolist()
            return normalized if isinstance(normalized, list) else list(normalized)
        except Exception:
            return []

    try:
        return list(value)
    except Exception:
        return []


def _format_cleanup_warnings(
    warnings: list[dict],
    skipped_file_deletions: Optional[list[dict]] = None,
) -> Optional[dict]:
    payload: dict = {}

    if warnings:
        payload["message"] = "Some cleanup steps failed"
        payload["errors"] = warnings

    if skipped_file_deletions:
        payload["skipped_file_deletions_count"] = len(skipped_file_deletions)
        payload["skipped_file_deletions_sample"] = skipped_file_deletions[:10]

    return payload or None


def _cleanup_warning(stage: str, target: str, error: Exception | str) -> dict:
    return {
        "stage": stage,
        "target": target,
        "error": str(error),
    }


def _collect_external_references(file_id: str, current_knowledge_id: str) -> dict:
    linked_knowledges = Knowledges.get_knowledges_by_file_id(file_id)
    other_knowledges = [k for k in linked_knowledges if k.id != current_knowledge_id]
    other_chats_count = Chats.count_chat_files_by_file_id(file_id)

    return {
        "other_collections_count": len(other_knowledges),
        "other_chats_count": other_chats_count,
        "other_collections_sample": [
            {"id": knowledge.id, "name": knowledge.name}
            for knowledge in other_knowledges[:10]
        ],
    }


def _get_other_collection_references(file_id: str, current_knowledge_id: str) -> list:
    linked_knowledges = Knowledges.get_knowledges_by_file_id(file_id)
    return [knowledge for knowledge in linked_knowledges if knowledge.id != current_knowledge_id]


def _has_other_collection_references(file_id: str, current_knowledge_id: str) -> bool:
    return len(_get_other_collection_references(file_id, current_knowledge_id)) > 0


def _build_skip_entry(file_id: str, references: list) -> dict:
    return {
        "file_id": file_id,
        "other_collections_count": len(references),
        "other_collections_sample": [
            {"id": knowledge.id, "name": knowledge.name}
            for knowledge in references[:10]
        ],
    }


def _detach_file_from_collection(knowledge, file_id: str) -> list[dict]:
    warnings: list[dict] = []

    collection_name = get_active_vector_collection_name(knowledge.id, knowledge.meta)
    file = Files.get_file_by_id(file_id)

    if collection_name:
        try:
            VECTOR_DB_CLIENT.delete(
                collection_name=collection_name,
                filter={"file_id": file_id},
            )
        except Exception as e:
            warnings.append(
                _cleanup_warning("vector.detach_by_file_id", collection_name, e)
            )

        if file and file.hash:
            try:
                VECTOR_DB_CLIENT.delete(
                    collection_name=collection_name,
                    filter={"hash": file.hash},
                )
            except Exception as e:
                warnings.append(
                    _cleanup_warning("vector.detach_by_hash", collection_name, e)
                )

    try:
        result = Knowledges.remove_file_from_knowledge_by_id(knowledge.id, file_id)
        if not result:
            warnings.append(_cleanup_warning("db.detach_file", file_id, "detach failed"))
    except Exception as e:
        warnings.append(_cleanup_warning("db.detach_file", file_id, e))

    if collection_name:
        try:
            mark_bm25_collections_dirty([collection_name])
            invalidate_bm25_collections([collection_name])
        except Exception as e:
            warnings.append(_cleanup_warning("bm25.invalidate", collection_name, e))

    return warnings


def _delete_file_entity(file_id: str) -> list[dict]:
    warnings: list[dict] = []

    file = Files.get_file_by_id(file_id)
    if not file:
        return warnings

    file_collection_name = f"file-{file_id}"
    try:
        if VECTOR_DB_CLIENT.has_collection(collection_name=file_collection_name):
            VECTOR_DB_CLIENT.delete_collection(collection_name=file_collection_name)
    except Exception as e:
        warnings.append(
            _cleanup_warning("vector.delete_file_collection", file_collection_name, e)
        )

    if file.path:
        try:
            Storage.delete_file(file.path)
        except Exception as e:
            warnings.append(_cleanup_warning("storage.delete_file", file.path, e))

    try:
        result = Files.delete_file_by_id(file_id)
        if not result:
            warnings.append(_cleanup_warning("db.delete_file", file_id, "delete failed"))
    except Exception as e:
        warnings.append(_cleanup_warning("db.delete_file", file_id, e))

    return warnings


def _hard_delete_file_everywhere(file_id: str) -> list[dict]:
    warnings: list[dict] = []

    file = Files.get_file_by_id(file_id)
    if not file:
        return warnings

    linked_knowledges = Knowledges.get_knowledges_by_file_id(file_id)
    touched_collections: set[str] = set()

    for knowledge in linked_knowledges:
        collection_name = get_active_vector_collection_name(knowledge.id, knowledge.meta)
        if not collection_name:
            continue

        touched_collections.add(collection_name)

        try:
            VECTOR_DB_CLIENT.delete(
                collection_name=collection_name,
                filter={"file_id": file_id},
            )
        except Exception as e:
            warnings.append(
                _cleanup_warning("vector.delete_by_file_id", collection_name, e)
            )

        if file.hash:
            try:
                VECTOR_DB_CLIENT.delete(
                    collection_name=collection_name,
                    filter={"hash": file.hash},
                )
            except Exception as e:
                warnings.append(
                    _cleanup_warning("vector.delete_by_hash", collection_name, e)
                )

    file_collection_name = f"file-{file_id}"
    try:
        if VECTOR_DB_CLIENT.has_collection(collection_name=file_collection_name):
            VECTOR_DB_CLIENT.delete_collection(collection_name=file_collection_name)
    except Exception as e:
        warnings.append(
            _cleanup_warning("vector.delete_file_collection", file_collection_name, e)
        )

    if touched_collections:
        try:
            collections = list(touched_collections)
            mark_bm25_collections_dirty(collections)
            invalidate_bm25_collections(collections)
        except Exception as e:
            warnings.append(_cleanup_warning("bm25.invalidate", file_id, e))

    if file.path:
        try:
            Storage.delete_file(file.path)
        except Exception as e:
            warnings.append(_cleanup_warning("storage.delete_file", file.path, e))

    try:
        result = Files.delete_file_by_id(file_id)
        if not result:
            warnings.append(_cleanup_warning("db.delete_file", file_id, "delete failed"))
    except Exception as e:
        warnings.append(_cleanup_warning("db.delete_file", file_id, e))

    return warnings


def _clone_qdrant_collection_vectors(
    source_collection_name: str,
    target_collection_name: str,
) -> tuple[bool, Optional[str]]:
    if VECTOR_DB != "qdrant":
        return False, "vector backend is not qdrant"

    if ENABLE_QDRANT_MULTITENANCY_MODE:
        return False, "qdrant multitenancy direct clone is not supported"

    client = getattr(VECTOR_DB_CLIENT, "client", None)
    collection_prefix = getattr(VECTOR_DB_CLIENT, "collection_prefix", None)

    if client is None or not collection_prefix:
        return False, "qdrant client is unavailable"

    source_collection = f"{collection_prefix}_{source_collection_name}"
    if not client.collection_exists(collection_name=source_collection):
        return False, "source vector collection does not exist"

    points = []
    offset = None

    while True:
        batch, offset = client.scroll(
            collection_name=source_collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        points.extend(batch)

        if offset is None:
            break

    if not points:
        return True, None

    items = []
    for point in points:
        vector = _extract_qdrant_vector(point.vector)
        if vector is None:
            return False, "source qdrant point is missing vector"

        payload = point.payload or {}
        metadata = payload.get("metadata") or {}
        items.append(
            {
                "id": point.id,
                "text": payload.get("text") or metadata.get("text") or "",
                "vector": vector,
                "metadata": metadata,
            }
        )

    for batch in _chunk_vector_items(items):
        VECTOR_DB_CLIENT.upsert(collection_name=target_collection_name, items=batch)

    return True, None


def _clone_chroma_by_file_id(
    source,
    target_collection_name: str,
    file_ids: list[str],
) -> dict:
    copied_count = 0
    missing_file_ids: list[str] = []
    failed_file_ids: list[str] = []
    missing_chunks = 0
    retry_success = 0
    retry_failed = 0

    for file_id in file_ids:
        result = None
        for attempt in range(1, FILE_ID_RETRY_ATTEMPTS + 1):
            try:
                result = source.get(
                    where={"file_id": file_id},
                    include=["embeddings", "documents", "metadatas"],
                )
                break
            except Exception as e:
                if attempt == FILE_ID_RETRY_ATTEMPTS:
                    result = None
                else:
                    delay = min(
                        FILE_ID_RETRY_BACKOFF_CAP,
                        FILE_ID_RETRY_BACKOFF_BASE * (2 ** (attempt - 1)),
                    )
                    time.sleep(delay)

        if result is None:
            failed_file_ids.append(file_id)
            retry_failed += 1
            continue

        retry_success += 1
        ids = _normalize_chroma_sequence(result.get("ids"))
        embeddings = _normalize_chroma_sequence(result.get("embeddings"))
        documents = _normalize_chroma_sequence(result.get("documents"))
        metadatas = _normalize_chroma_sequence(result.get("metadatas"))

        if len(ids) == 0:
            missing_file_ids.append(file_id)
            continue

        copy_ids = []
        copy_embeddings = []
        copy_documents = []
        copy_metadatas = []

        for i, item_id in enumerate(ids):
            embedding = embeddings[i] if i < len(embeddings) else None
            document = documents[i] if i < len(documents) else ""
            metadata = metadatas[i] if i < len(metadatas) else {}

            if embedding is None:
                missing_chunks += 1
                continue

            copy_ids.append(item_id)
            copy_embeddings.append(embedding)
            copy_documents.append("" if document is None else document)
            copy_metadatas.append(metadata if isinstance(metadata, dict) else {})

        if not copy_ids:
            missing_file_ids.append(file_id)
            continue

        for i in range(0, len(copy_ids), 256):
            _upsert_chroma_records(
                target_collection_name=target_collection_name,
                ids=copy_ids[i : i + 256],
                embeddings=copy_embeddings[i : i + 256],
                documents=copy_documents[i : i + 256],
                metadatas=copy_metadatas[i : i + 256],
            )

        copied_count += len(copy_ids)

    return {
        "copied_count": copied_count,
        "missing_file_ids": missing_file_ids,
        "failed_file_ids": failed_file_ids,
        "missing_chunks": missing_chunks,
        "status_detail": "retry_by_file_id",
        "file_id_retry_attempts": FILE_ID_RETRY_ATTEMPTS,
        "file_id_retry_success_count": retry_success,
        "file_id_retry_failed_count": retry_failed,
    }


def _clone_chroma_collection_vectors(
    source_collection_name: str,
    target_collection_name: str,
    file_ids: Optional[list[str]] = None,
) -> dict:
    defaults = {
        "copied_count": 0,
        "missing_file_ids": [],
        "failed_file_ids": [],
        "file_id_retry_attempts": None,
        "file_id_success_count": None,
        "file_id_failed_count": None,
        "has_unknown_missing": False,
        "total_items": 0,
        "page_size": CHROMA_CLONE_PAGE_SIZE,
        "pages_scanned": 0,
        "failed_offset": None,
        "failed_stage": None,
        "missing_chunks": 0,
        "status_detail": None,
        "retry_mode": None,
    }

    if VECTOR_DB != "chroma":
        return {
            **defaults,
            "status": "failed",
            "reason": "vector backend is not chroma",
            "failed_stage": "backend",
            "status_detail": "wrong_backend",
        }

    client = getattr(VECTOR_DB_CLIENT, "client", None)
    if client is None:
        return {
            **defaults,
            "status": "failed",
            "reason": "chroma client is unavailable",
            "failed_stage": "client",
            "status_detail": "client_unavailable",
        }

    try:
        source = client.get_collection(name=source_collection_name)
    except Exception:
        return {
            **defaults,
            "status": "failed",
            "reason": "source vector collection does not exist",
            "failed_stage": "collection",
            "status_detail": "collection_not_found",
        }

    try:
        total_items = int(source.count())
    except Exception as e:
        return {
            **defaults,
            "status": "failed",
            "reason": f"failed to count source chroma collection: {e}",
            "total_items": 0,
            "failed_stage": "count",
            "status_detail": "count_failed",
        }

    if total_items == 0:
        return {
            **defaults,
            "status": "success",
            "reason": None,
            "total_items": 0,
            "status_detail": "empty_collection",
        }

    copied_count = 0
    missing_file_ids = set()
    has_unknown_missing = False
    missing_chunks = 0
    pages_scanned = 0

    offset = 0
    while offset < total_items:
        try:
            result = source.get(
                limit=CHROMA_CLONE_PAGE_SIZE,
                offset=offset,
                include=["embeddings", "documents", "metadatas"],
            )
        except Exception as e:
            if file_ids:
                retry_result = _clone_chroma_by_file_id(
                    source,
                    target_collection_name,
                    file_ids,
                )
                failed_file_ids = retry_result["failed_file_ids"]
                missing_file_ids = set(retry_result["missing_file_ids"])
                missing_chunks = retry_result["missing_chunks"]
                copied_count = retry_result["copied_count"]

                failed_preview = failed_file_ids[:20]
                return {
                    "status": "partial" if copied_count > 0 else "failed",
                    "reason": f"file_id retry after paged get failed at offset {offset}: {e}",
                    "copied_count": copied_count,
                    "missing_file_ids": list(missing_file_ids),
                    "failed_file_ids": failed_file_ids,
                    "has_unknown_missing": False,
                    "total_items": total_items,
                    "page_size": CHROMA_CLONE_PAGE_SIZE,
                    "pages_scanned": pages_scanned,
                    "failed_offset": offset,
                    "failed_stage": "get_page",
                    "missing_chunks": missing_chunks,
                    "status_detail": retry_result["status_detail"],
                    "retry_mode": "file_id",
                    "file_id_retry_attempts": retry_result.get("file_id_retry_attempts"),
                    "file_id_success_count": retry_result.get("file_id_retry_success_count"),
                    "file_id_failed_count": retry_result.get("file_id_retry_failed_count"),
                    "failed_file_ids": failed_preview,
                }

            return {
                "status": "partial" if copied_count > 0 else "failed",
                "reason": f"failed to read source chroma collection at offset {offset}: {e}",
                "copied_count": copied_count,
                "missing_file_ids": list(missing_file_ids),
                "failed_file_ids": [],
                "has_unknown_missing": True,
                "total_items": total_items,
                "page_size": CHROMA_CLONE_PAGE_SIZE,
                "pages_scanned": pages_scanned,
                "failed_offset": offset,
                "failed_stage": "get_page",
                "missing_chunks": missing_chunks,
                "status_detail": "paged_get_failed",
                "retry_mode": None,
            }

        pages_scanned += 1

        ids = _normalize_chroma_sequence(result.get("ids"))
        embeddings = _normalize_chroma_sequence(result.get("embeddings"))
        documents = _normalize_chroma_sequence(result.get("documents"))
        metadatas = _normalize_chroma_sequence(result.get("metadatas"))

        copy_ids = []
        copy_embeddings = []
        copy_documents = []
        copy_metadatas = []

        for i, item_id in enumerate(ids):
            embedding = embeddings[i] if i < len(embeddings) else None
            document = documents[i] if i < len(documents) else ""
            metadata = metadatas[i] if i < len(metadatas) else {}

            if embedding is None:
                missing_chunks += 1
                if isinstance(metadata, dict) and metadata.get("file_id"):
                    missing_file_ids.add(metadata["file_id"])
                else:
                    has_unknown_missing = True
                continue

            copy_ids.append(item_id)
            copy_embeddings.append(embedding)
            copy_documents.append("" if document is None else document)
            copy_metadatas.append(metadata if isinstance(metadata, dict) else {})

        for i in range(0, len(copy_ids), 256):
            try:
                _upsert_chroma_records(
                    target_collection_name=target_collection_name,
                    ids=copy_ids[i : i + 256],
                    embeddings=copy_embeddings[i : i + 256],
                    documents=copy_documents[i : i + 256],
                    metadatas=copy_metadatas[i : i + 256],
                )
            except Exception as e:
                return {
                    "status": "partial" if copied_count > 0 else "failed",
                    "reason": f"failed to upsert target chroma collection at offset {offset}: {e}",
                    "copied_count": copied_count,
                    "missing_file_ids": list(missing_file_ids),
                    "has_unknown_missing": True,
                    "total_items": total_items,
                    "page_size": CHROMA_CLONE_PAGE_SIZE,
                    "pages_scanned": pages_scanned,
                    "failed_offset": offset,
                    "failed_stage": "upsert",
                    "missing_chunks": missing_chunks,
                    "status_detail": "upsert_failed",
                }

        copied_count += len(copy_ids)

        # Defensive progress step to avoid infinite loops on unexpected empty pages.
        if len(ids) == 0:
            offset += CHROMA_CLONE_PAGE_SIZE
        else:
            offset += len(ids)

    if copied_count == total_items and not has_unknown_missing and len(missing_file_ids) == 0:
        status = "success"
        reason = None
        status_detail = "all_chunks_copied"
    else:
        status = "partial"
        status_detail = "incomplete_embeddings"
        reason = (
            f"incomplete embeddings during clone: missing_chunks={missing_chunks}, "
            f"missing_file_ids={len(missing_file_ids)}"
        )

    return {
        "status": status,
        "reason": reason,
        "copied_count": copied_count,
        "missing_file_ids": list(missing_file_ids),
        "failed_file_ids": [],
        "has_unknown_missing": has_unknown_missing,
        "total_items": total_items,
        "page_size": CHROMA_CLONE_PAGE_SIZE,
        "pages_scanned": pages_scanned,
        "failed_offset": None,
        "failed_stage": None,
        "missing_chunks": missing_chunks,
        "status_detail": status_detail,
        "retry_mode": None,
    }


async def _clone_collection_vectors(
    request: Request,
    source_knowledge,
    target_knowledge,
    files: List[FileModel],
    user,
) -> dict:
    source_collection_name = get_active_vector_collection_name(
        source_knowledge.id, source_knowledge.meta
    )
    target_collection_name = get_active_vector_collection_name(
        target_knowledge.id, target_knowledge.meta
    )

    fallback_reason = "source vector collection does not exist"
    direct_clone_success = False
    copied_count = 0
    missing_file_ids: list[str] = []
    has_unknown_missing = False
    chroma_observability = {
        "total_items": None,
        "pages_scanned": None,
        "failed_offset": None,
        "failed_stage": None,
        "missing_chunks": None,
        "status_detail": None,
        "page_size": None,
        "retry_mode": None,
        "failed_file_ids": [],
        "file_id_retry_attempts": None,
        "file_id_success_count": None,
        "file_id_failed_count": None,
    }

    log.info(
        "Knowledge clone start id=%s vector_db=%s source_collection=%s target_collection=%s file_count=%d chroma_page_size=%s",
        source_knowledge.id,
        VECTOR_DB,
        source_collection_name,
        target_collection_name,
        len(files),
        CHROMA_CLONE_PAGE_SIZE if VECTOR_DB == "chroma" else "n/a",
    )

    if VECTOR_DB_CLIENT.has_collection(collection_name=source_collection_name):
        try:
            if VECTOR_DB == "qdrant":
                direct_clone_success, fallback_reason = await run_in_threadpool(
                    _clone_qdrant_collection_vectors,
                    source_collection_name=source_collection_name,
                    target_collection_name=target_collection_name,
                )
            elif VECTOR_DB == "chroma":
                chroma_clone_result = await run_in_threadpool(
                    _clone_chroma_collection_vectors,
                    source_collection_name=source_collection_name,
                    target_collection_name=target_collection_name,
                    file_ids=[file.id for file in files],
                )
                chroma_observability = {
                    "total_items": chroma_clone_result.get("total_items"),
                    "pages_scanned": chroma_clone_result.get("pages_scanned"),
                    "failed_offset": chroma_clone_result.get("failed_offset"),
                    "failed_stage": chroma_clone_result.get("failed_stage"),
                    "missing_chunks": chroma_clone_result.get("missing_chunks"),
                    "status_detail": chroma_clone_result.get("status_detail"),
                    "page_size": chroma_clone_result.get("page_size"),
                    "retry_mode": chroma_clone_result.get("retry_mode"),
                    "failed_file_ids": chroma_clone_result.get("failed_file_ids") or [],
                    "file_id_retry_attempts": chroma_clone_result.get("file_id_retry_attempts"),
                    "file_id_success_count": chroma_clone_result.get("file_id_retry_success_count") or chroma_clone_result.get("file_id_success_count"),
                    "file_id_failed_count": chroma_clone_result.get("file_id_retry_failed_count") or chroma_clone_result.get("file_id_failed_count"),
                }
                direct_clone_success = chroma_clone_result["status"] == "success"
                copied_count = chroma_clone_result["copied_count"]
                missing_file_ids = chroma_clone_result["missing_file_ids"]
                has_unknown_missing = chroma_clone_result["has_unknown_missing"]
                fallback_reason = chroma_clone_result["reason"] or fallback_reason
                if chroma_clone_result.get("retry_mode") == "file_id":
                    missing_file_ids = chroma_clone_result.get("failed_file_ids") or []
                    has_unknown_missing = False

                log.info(
                    "Knowledge direct clone result id=%s status=%s copied_count=%d total_items=%s pages_scanned=%s missing_file_ids=%d missing_chunks=%s failed_stage=%s failed_offset=%s status_detail=%s retry_mode=%s failed_file_ids=%d",
                    source_knowledge.id,
                    chroma_clone_result.get("status"),
                    copied_count,
                    chroma_observability["total_items"],
                    chroma_observability["pages_scanned"],
                    len(missing_file_ids),
                    chroma_observability["missing_chunks"],
                    chroma_observability["failed_stage"],
                    chroma_observability["failed_offset"],
                    chroma_observability["status_detail"],
                    chroma_observability["retry_mode"],
                    len(chroma_observability["failed_file_ids"]),
                )
            else:
                fallback_reason = (
                    f"vector direct clone is not implemented for backend {VECTOR_DB}"
                )
        except Exception as e:
            log.exception(f"Direct vector clone failed for knowledge {source_knowledge.id}: {e}")
            fallback_reason = str(e)

    if direct_clone_success:
        log.info(
            "Knowledge clone completed with direct copy id=%s copied_count=%d total_items=%s pages_scanned=%s",
            source_knowledge.id,
            copied_count,
            chroma_observability["total_items"],
            chroma_observability["pages_scanned"],
        )
        return {
            "successful_file_ids": [file.id for file in files],
            "warnings": {
                "strategy": "direct_copy",
                "copied_count": copied_count,
                    "total_items": chroma_observability["total_items"],
                    "pages_scanned": chroma_observability["pages_scanned"],
                    "failed_offset": chroma_observability["failed_offset"],
                    "failed_stage": chroma_observability["failed_stage"],
                    "missing_chunks": chroma_observability["missing_chunks"],
                    "retry_mode": chroma_observability["retry_mode"],
                    "file_id_success_count": chroma_observability["file_id_success_count"],
                    "file_id_failed_count": chroma_observability["file_id_failed_count"],
                    "failed_file_ids": chroma_observability["failed_file_ids"],
                },
        }

    files_to_reembed = files
    strategy = "full_reembed_fallback"

    if VECTOR_DB == "chroma" and copied_count > 0 and not has_unknown_missing:
        missing_file_id_set = set(missing_file_ids)
        if missing_file_id_set:
            files_to_reembed = [file for file in files if file.id in missing_file_id_set]
            if files_to_reembed:
                strategy = "partial_copy_with_reembed"
            else:
                strategy = "direct_copy"
        else:
            strategy = "direct_copy" if chroma_observability["retry_mode"] == "file_id" else "full_reembed_fallback"

    if strategy == "direct_copy":
        log.info(
            "Knowledge clone resolved to direct copy id=%s copied_count=%d total_items=%s pages_scanned=%s",
            source_knowledge.id,
            copied_count,
            chroma_observability["total_items"],
            chroma_observability["pages_scanned"],
        )
        return {
            "successful_file_ids": [file.id for file in files],
            "warnings": {
                "strategy": "direct_copy",
                "copied_count": copied_count,
                    "total_items": chroma_observability["total_items"],
                    "pages_scanned": chroma_observability["pages_scanned"],
                    "failed_offset": chroma_observability["failed_offset"],
                    "failed_stage": chroma_observability["failed_stage"],
                    "missing_chunks": chroma_observability["missing_chunks"],
                    "retry_mode": chroma_observability["retry_mode"],
                    "file_id_success_count": chroma_observability["file_id_success_count"],
                    "file_id_failed_count": chroma_observability["file_id_failed_count"],
                    "failed_file_ids": chroma_observability["failed_file_ids"],
                },
        }

    log.warning(
        "Knowledge clone fallback id=%s strategy=%s reason=%s copied_count=%d total_items=%s pages_scanned=%s missing_file_ids=%d unknown_missing=%s failed_stage=%s failed_offset=%s retry_mode=%s file_id_failed=%s",
        source_knowledge.id,
        strategy,
        fallback_reason,
        copied_count,
        chroma_observability["total_items"],
        chroma_observability["pages_scanned"],
        len(missing_file_ids),
        has_unknown_missing,
        chroma_observability["failed_stage"],
        chroma_observability["failed_offset"],
        chroma_observability["retry_mode"],
        chroma_observability["file_id_failed_count"],
    )

    if strategy != "partial_copy_with_reembed":
        try:
            if VECTOR_DB_CLIENT.has_collection(collection_name=target_collection_name):
                await run_in_threadpool(
                    VECTOR_DB_CLIENT.delete_collection,
                    collection_name=target_collection_name,
                )
        except Exception:
            pass

    successful_file_ids = []
    fallback_errors = []
    reembedded_file_ids = []

    if files_to_reembed:
        fallback_result = await process_files_batch(
            request=request,
            form_data=BatchProcessFilesForm(
                files=files_to_reembed,
                collection_name=target_collection_name,
            ),
            user=user,
        )

        successful_file_ids = [
            result.file_id
            for result in fallback_result.results
            if result.status == "completed"
        ]
        fallback_errors = [
            {"file_id": error.file_id, "error": error.error}
            for error in fallback_result.errors
        ]

        reembedded_file_ids = [
            result.file_id
            for result in fallback_result.results
            if result.status == "completed"
        ]

    if strategy == "partial_copy_with_reembed":
        missing_file_id_set = set(missing_file_ids)
        copied_file_ids = [
            file.id for file in files if file.id not in missing_file_id_set
        ]
        successful_file_ids = copied_file_ids + successful_file_ids

    warnings = {
        "message": "vector direct clone unavailable, fallback to re-embed",
        "reason": fallback_reason,
        "strategy": strategy,
        "copied_count": copied_count,
        "reembedded_file_ids": reembedded_file_ids,
        "total_items": chroma_observability["total_items"],
        "pages_scanned": chroma_observability["pages_scanned"],
        "failed_offset": chroma_observability["failed_offset"],
        "failed_stage": chroma_observability["failed_stage"],
        "missing_chunks": chroma_observability["missing_chunks"],
        "status_detail": chroma_observability["status_detail"],
        "retry_mode": chroma_observability["retry_mode"],
        "file_id_retry_attempts": chroma_observability.get("file_id_retry_attempts"),
        "file_id_success_count": chroma_observability["file_id_success_count"],
        "file_id_failed_count": chroma_observability["file_id_failed_count"],
        "failed_file_ids": chroma_observability["failed_file_ids"],
    }
    if fallback_errors:
        warnings["errors"] = fallback_errors

    log.info(
        "Knowledge clone fallback completed id=%s strategy=%s reembedded_file_count=%d fallback_error_count=%d",
        source_knowledge.id,
        strategy,
        len(reembedded_file_ids),
        len(fallback_errors),
    )

    return {
        "successful_file_ids": successful_file_ids,
        "warnings": warnings,
    }


async def _reindex_collection_internal(request: Request, knowledge, user):
    active_collection_name = get_active_vector_collection_name(knowledge.id, knowledge.meta)
    temp_collection_name = f"{knowledge.id}__reindex_tmp__{uuid.uuid4().hex[:8]}"

    pending_meta = upsert_reindex_status_meta(
        knowledge.meta,
        active_collection_name=active_collection_name,
        status="running",
        success=False,
    )
    Knowledges.update_knowledge_by_id(
        id=knowledge.id,
        form_data=KnowledgeForm(meta=pending_meta),
    )

    files = Knowledges.get_files_by_id(knowledge.id)
    failed_files = []

    try:
        try:
            if VECTOR_DB_CLIENT.has_collection(collection_name=temp_collection_name):
                await run_in_threadpool(
                    VECTOR_DB_CLIENT.delete_collection,
                    collection_name=temp_collection_name,
                )
        except Exception:
            pass

        for file in files:
            try:
                await run_in_threadpool(
                    process_file,
                    request,
                    ProcessFileForm(
                        file_id=file.id,
                        collection_name=temp_collection_name,
                        knowledge_id=knowledge.id,
                    ),
                    user=user,
                )
            except Exception as e:
                failed_files.append({"file_id": file.id, "error": str(e)})

        if failed_files:
            raise Exception("Reindex failed for one or more files")

        next_meta = upsert_reindex_status_meta(
            knowledge.meta,
            active_collection_name=temp_collection_name,
            status="success",
            success=True,
        )
        Knowledges.update_knowledge_by_id(
            id=knowledge.id,
            form_data=KnowledgeForm(meta=next_meta),
        )

        if active_collection_name != temp_collection_name:
            try:
                if VECTOR_DB_CLIENT.has_collection(collection_name=active_collection_name):
                    await run_in_threadpool(
                        VECTOR_DB_CLIENT.delete_collection,
                        collection_name=active_collection_name,
                    )
            except Exception as e:
                log.warning(
                    f"Failed to delete old collection {active_collection_name} after successful reindex: {e}"
                )

        return {
            "status": True,
            "collection_name": knowledge.id,
            "active_collection_name": temp_collection_name,
            "total_files": len(files),
            "failed_files": failed_files,
        }
    except Exception as e:
        try:
            if VECTOR_DB_CLIENT.has_collection(collection_name=temp_collection_name):
                await run_in_threadpool(
                    VECTOR_DB_CLIENT.delete_collection,
                    collection_name=temp_collection_name,
                )
        except Exception:
            pass

        failed_meta = upsert_reindex_status_meta(
            knowledge.meta,
            active_collection_name=active_collection_name,
            status="failed",
            success=False,
        )
        Knowledges.update_knowledge_by_id(
            id=knowledge.id,
            form_data=KnowledgeForm(meta=failed_meta),
        )

        log.exception(f"Collection reindex failed for {knowledge.id}: {e}")
        return {
            "status": False,
            "collection_name": knowledge.id,
            "active_collection_name": active_collection_name,
            "total_files": len(files),
            "failed_files": failed_files,
            "error": str(e),
        }


@router.get("/", response_model=KnowledgeAccessListResponse)
async def get_knowledge_bases(page: Optional[int] = 1, user=Depends(get_verified_user)):
    page = max(page, 1)
    limit = PAGE_ITEM_COUNT
    skip = (page - 1) * limit

    filter = {}
    if not user.role == "admin" or not BYPASS_ADMIN_ACCESS_CONTROL:
        groups = Groups.get_groups_by_member_id(user.id)
        if groups:
            filter["group_ids"] = [group.id for group in groups]

        filter["user_id"] = user.id

    result = Knowledges.search_knowledge_bases(
        user.id, filter=filter, skip=skip, limit=limit
    )

    return KnowledgeAccessListResponse(
        items=[
            KnowledgeAccessResponse(
                **knowledge_base.model_dump(),
                write_access=(
                    user.id == knowledge_base.user_id
                    or has_access(user.id, "write", knowledge_base.access_control)
                ),
            )
            for knowledge_base in result.items
        ],
        total=result.total,
    )


@router.get("/search", response_model=KnowledgeAccessListResponse)
async def search_knowledge_bases(
    query: Optional[str] = None,
    view_option: Optional[str] = None,
    page: Optional[int] = 1,
    user=Depends(get_verified_user),
):
    page = max(page, 1)
    limit = PAGE_ITEM_COUNT
    skip = (page - 1) * limit

    filter = {}
    if query:
        filter["query"] = query
    if view_option:
        filter["view_option"] = view_option

    if not user.role == "admin" or not BYPASS_ADMIN_ACCESS_CONTROL:
        groups = Groups.get_groups_by_member_id(user.id)
        if groups:
            filter["group_ids"] = [group.id for group in groups]

        filter["user_id"] = user.id

    result = Knowledges.search_knowledge_bases(
        user.id, filter=filter, skip=skip, limit=limit
    )

    return KnowledgeAccessListResponse(
        items=[
            KnowledgeAccessResponse(
                **knowledge_base.model_dump(),
                write_access=(
                    user.id == knowledge_base.user_id
                    or has_access(user.id, "write", knowledge_base.access_control)
                ),
            )
            for knowledge_base in result.items
        ],
        total=result.total,
    )


@router.get("/search/files", response_model=KnowledgeFileListResponse)
async def search_knowledge_files(
    query: Optional[str] = None,
    page: Optional[int] = 1,
    user=Depends(get_verified_user),
):
    page = max(page, 1)
    limit = PAGE_ITEM_COUNT
    skip = (page - 1) * limit

    filter = {}
    if query:
        filter["query"] = query

    groups = Groups.get_groups_by_member_id(user.id)
    if groups:
        filter["group_ids"] = [group.id for group in groups]

    filter["user_id"] = user.id

    return Knowledges.search_knowledge_files(filter=filter, skip=skip, limit=limit)


############################
# CreateNewKnowledge
############################


@router.post("/create", response_model=Optional[KnowledgeResponse])
async def create_new_knowledge(
    request: Request, form_data: KnowledgeForm, user=Depends(get_verified_user)
):
    _ensure_knowledge_create_permission(request, user)

    # Check if user can share publicly
    if (
        user.role != "admin"
        and form_data.access_control == None
        and not has_permission(
            user.id,
            "sharing.public_knowledge",
            request.app.state.config.USER_PERMISSIONS,
        )
    ):
        form_data.access_control = {}

    if not (form_data.name and form_data.name.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Knowledge name is required",
        )

    if form_data.description is None:
        form_data.description = ""

    knowledge = Knowledges.insert_new_knowledge(user.id, form_data)

    if knowledge:
        return knowledge
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.FILE_EXISTS,
        )


############################
# ReindexKnowledgeFiles
############################


@router.post("/reindex", response_model=bool)
async def reindex_knowledge_files(request: Request, user=Depends(get_verified_user)):
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    knowledge_bases = Knowledges.get_knowledge_bases()

    log.info(f"Starting reindexing for {len(knowledge_bases)} knowledge bases")

    for knowledge_base in knowledge_bases:
        result = await _reindex_collection_internal(request, knowledge_base, user)
        if not result.get("status"):
            log.warning(
                f"Collection reindex failed for {knowledge_base.id}: {result.get('error', 'unknown error')}"
            )

    log.info(f"Reindexing completed.")
    return True


@router.get("/{id}/config")
async def get_knowledge_config_by_id(id: str, request: Request, user=Depends(get_verified_user)):
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    _ensure_write_access(knowledge, user)

    config_info = resolve_collection_rag_config(knowledge.meta, request.app.state.config)
    active_collection_name = get_active_vector_collection_name(knowledge.id, knowledge.meta)

    vector_meta = (knowledge.meta or {}).get("collection_vector") or {}

    return {
        "status": True,
        "mode": config_info["mode"],
        "overrides": config_info["overrides"],
        "global_defaults": config_info["global_defaults"],
        "effective": config_info["effective"],
        "vector": {
            "active_collection_name": active_collection_name,
            "reindex_version": int(vector_meta.get("reindex_version") or 0),
            "last_reindexed_at": vector_meta.get("last_reindexed_at"),
            "last_reindex_status": vector_meta.get("last_reindex_status"),
        },
    }


@router.post("/{id}/config")
async def update_knowledge_config_by_id(
    id: str,
    request: Request,
    form_data: CollectionRagConfigUpdateForm,
    user=Depends(get_verified_user),
):
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    _ensure_write_access(knowledge, user)

    mode = form_data.mode if form_data.mode in {"default", "custom"} else "default"
    overrides = sanitize_rag_overrides(form_data.overrides)
    _validate_collection_rag_overrides(overrides)

    next_meta = upsert_collection_rag_meta(
        knowledge.meta,
        mode=mode,
        overrides=overrides,
    )

    updated = Knowledges.update_knowledge_by_id(
        id=id,
        form_data=KnowledgeForm(meta=next_meta),
    )

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT("Failed to update collection config"),
        )

    config_info = resolve_collection_rag_config(updated.meta, request.app.state.config)

    return {
        "status": True,
        "mode": config_info["mode"],
        "overrides": config_info["overrides"],
        "effective": config_info["effective"],
    }


@router.post("/{id}/reindex")
async def reindex_knowledge_by_id(id: str, request: Request, user=Depends(get_verified_user)):
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    _ensure_write_access(knowledge, user)
    return await _reindex_collection_internal(request, knowledge, user)


############################
# GetKnowledgeById
############################


class KnowledgeFilesResponse(KnowledgeResponse):
    files: Optional[list[FileMetadataResponse]] = None
    write_access: Optional[bool] = False


@router.get("/{id}", response_model=Optional[KnowledgeFilesResponse])
async def get_knowledge_by_id(id: str, user=Depends(get_verified_user)):
    knowledge = Knowledges.get_knowledge_by_id(id=id)

    if knowledge:
        if (
            user.role == "admin"
            or knowledge.user_id == user.id
            or has_access(user.id, "read", knowledge.access_control)
        ):

            return KnowledgeFilesResponse(
                **knowledge.model_dump(),
                write_access=(
                    user.id == knowledge.user_id
                    or has_access(user.id, "write", knowledge.access_control)
                ),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# UpdateKnowledgeById
############################


@router.post("/{id}/update", response_model=Optional[KnowledgeFilesResponse])
async def update_knowledge_by_id(
    request: Request,
    id: str,
    form_data: KnowledgeForm,
    user=Depends(get_verified_user),
):
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )
    # Is the user the original creator, in a group with write access, or an admin
    if (
        knowledge.user_id != user.id
        and not has_access(user.id, "write", knowledge.access_control)
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    # Check if user can share publicly
    if (
        user.role != "admin"
        and form_data.access_control == None
        and not has_permission(
            user.id,
            "sharing.public_knowledge",
            request.app.state.config.USER_PERMISSIONS,
        )
    ):
        form_data.access_control = {}

    knowledge = Knowledges.update_knowledge_by_id(id=id, form_data=form_data)
    if knowledge:
        return KnowledgeFilesResponse(
            **knowledge.model_dump(),
            files=Knowledges.get_file_metadatas_by_id(knowledge.id),
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ID_TAKEN,
        )


############################
# GetKnowledgeFilesById
############################


@router.get("/{id}/files", response_model=KnowledgeFileListResponse)
async def get_knowledge_files_by_id(
    id: str,
    query: Optional[str] = None,
    view_option: Optional[str] = None,
    order_by: Optional[str] = None,
    direction: Optional[str] = None,
    page: Optional[int] = 1,
    user=Depends(get_verified_user),
):

    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if not (
        user.role == "admin"
        or knowledge.user_id == user.id
        or has_access(user.id, "read", knowledge.access_control)
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    page = max(page, 1)

    limit = 30
    skip = (page - 1) * limit

    filter = {}
    if query:
        filter["query"] = query
    if view_option:
        filter["view_option"] = view_option
    if order_by:
        filter["order_by"] = order_by
    if direction:
        filter["direction"] = direction

    return Knowledges.search_files_by_id(
        id, user.id, filter=filter, skip=skip, limit=limit
    )


@router.get("/{id}/download")
async def download_knowledge_by_id(
    id: str,
    user=Depends(get_verified_user),
):
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if not (
        user.role == "admin"
        or knowledge.user_id == user.id
        or has_access(user.id, "read", knowledge.access_control)
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    files = Knowledges.get_files_by_id(id)

    try:
        zip_path, _ = await run_in_threadpool(_build_knowledge_download_zip, knowledge, files)
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT("Failed to build collection archive"),
        )

    archive_name = _build_knowledge_archive_name(knowledge)
    encoded_filename = quote(archive_name)

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}",
        },
        background=BackgroundTask(_remove_temp_file, zip_path),
    )


############################
# AddFileToKnowledge
############################


class KnowledgeFileIdForm(BaseModel):
    file_id: str


@router.post("/{id}/file/add", response_model=Optional[KnowledgeFilesResponse])
def add_file_to_knowledge_by_id(
    request: Request,
    id: str,
    form_data: KnowledgeFileIdForm,
    user=Depends(get_verified_user),
):
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        knowledge.user_id != user.id
        and not has_access(user.id, "write", knowledge.access_control)
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    file = Files.get_file_by_id(form_data.file_id)
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )
    if not file.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.FILE_NOT_PROCESSED,
        )

    # Add content to the vector database
    try:
        collection_name = get_active_vector_collection_name(knowledge.id, knowledge.meta)
        process_file(
            request,
            ProcessFileForm(
                file_id=form_data.file_id,
                collection_name=collection_name,
                knowledge_id=knowledge.id,
            ),
            user=user,
        )

        # Add file to knowledge base
        Knowledges.add_file_to_knowledge_by_id(
            knowledge_id=id, file_id=form_data.file_id, user_id=user.id
        )
    except Exception as e:
        log.debug(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if knowledge:
        return KnowledgeFilesResponse(
            **knowledge.model_dump(),
            files=Knowledges.get_file_metadatas_by_id(knowledge.id),
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


@router.post("/{id}/file/update", response_model=Optional[KnowledgeFilesResponse])
async def update_file_from_knowledge_by_id(
    request: Request,
    id: str,
    form_data: KnowledgeFileIdForm,
    user=Depends(get_verified_user),
):
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        knowledge.user_id != user.id
        and not has_access(user.id, "write", knowledge.access_control)
        and user.role != "admin"
    ):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    file = Files.get_file_by_id(form_data.file_id)
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    # Remove content from the vector database
    collection_name = get_active_vector_collection_name(knowledge.id, knowledge.meta)
    await run_in_threadpool(
        VECTOR_DB_CLIENT.delete,
        collection_name=collection_name,
        filter={"file_id": form_data.file_id},
    )

    # Add content to the vector database
    try:
        await run_in_threadpool(
            process_file,
            request,
            ProcessFileForm(
                file_id=form_data.file_id,
                collection_name=collection_name,
                knowledge_id=knowledge.id,
            ),
            user=user,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if knowledge:
        return KnowledgeFilesResponse(
            **knowledge.model_dump(),
            files=Knowledges.get_file_metadatas_by_id(knowledge.id),
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# RemoveFileFromKnowledge
############################


@router.post("/{id}/file/remove", response_model=Optional[KnowledgeFilesResponse])
def remove_file_from_knowledge_by_id(
    id: str,
    form_data: KnowledgeFileIdForm,
    force: bool = Query(False),
    user=Depends(get_verified_user),
):
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        knowledge.user_id != user.id
        and not has_access(user.id, "write", knowledge.access_control)
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    file = Files.get_file_by_id(form_data.file_id)
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    warnings: list[dict] = []
    skipped_file_deletions: list[dict] = []

    warnings.extend(_detach_file_from_collection(knowledge, form_data.file_id))

    other_collection_references = _get_other_collection_references(form_data.file_id, id)
    if other_collection_references:
        skipped_file_deletions.append(
            _build_skip_entry(form_data.file_id, other_collection_references)
        )
    else:
        warnings.extend(_delete_file_entity(form_data.file_id))

    if knowledge:
        return KnowledgeFilesResponse(
            **knowledge.model_dump(),
            files=Knowledges.get_file_metadatas_by_id(knowledge.id),
            warnings=_format_cleanup_warnings(
                warnings,
                skipped_file_deletions=skipped_file_deletions,
            ),
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# CloneKnowledgeById
############################


@router.post("/{id}/clone", response_model=Optional[KnowledgeFilesResponse])
async def clone_knowledge_by_id(
    request: Request,
    id: str,
    user=Depends(get_verified_user),
):
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    _ensure_write_access(knowledge, user)
    _ensure_knowledge_create_permission(request, user)

    cloned_access_control = copy.deepcopy(knowledge.access_control)
    if (
        user.role != "admin"
        and cloned_access_control is None
        and not has_permission(
            user.id,
            "sharing.public_knowledge",
            request.app.state.config.USER_PERMISSIONS,
        )
    ):
        cloned_access_control = {}

    cloned_knowledge = Knowledges.insert_new_knowledge(
        user.id,
        KnowledgeForm(
            name=f"{knowledge.name}_Clone",
            description=knowledge.description,
            meta=_build_cloned_knowledge_meta(knowledge.meta),
            access_control=cloned_access_control,
        ),
    )

    if not cloned_knowledge:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT("Failed to clone knowledge base"),
        )

    files = Knowledges.get_files_by_id(knowledge.id)
    clone_result = await _clone_collection_vectors(
        request=request,
        source_knowledge=knowledge,
        target_knowledge=cloned_knowledge,
        files=files,
        user=user,
    )

    successful_file_ids = set(clone_result["successful_file_ids"])
    for file in files:
        if file.id not in successful_file_ids:
            continue

        Knowledges.add_file_to_knowledge_by_id(
            knowledge_id=cloned_knowledge.id,
            file_id=file.id,
            user_id=user.id,
        )

    return KnowledgeFilesResponse(
        **cloned_knowledge.model_dump(),
        files=Knowledges.get_file_metadatas_by_id(cloned_knowledge.id),
        warnings=clone_result["warnings"],
    )


############################
# DeleteKnowledgeById
############################


@router.delete("/{id}/delete", response_model=DeleteKnowledgeResponse)
async def delete_knowledge_by_id(
    id: str,
    force: bool = Query(False),
    user=Depends(get_verified_user),
):
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        knowledge.user_id != user.id
        and not has_access(user.id, "write", knowledge.access_control)
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    log.info(f"Deleting knowledge base: {id} (name: {knowledge.name})")

    files = Knowledges.get_files_by_id(id)

    warnings: list[dict] = []
    skipped_file_deletions: list[dict] = []

    seen_file_ids: set[str] = set()
    for file in files:
        if file.id in seen_file_ids:
            continue

        seen_file_ids.add(file.id)
        other_collection_references = _get_other_collection_references(file.id, id)
        if other_collection_references:
            skipped_file_deletions.append(
                _build_skip_entry(file.id, other_collection_references)
            )
            continue

        warnings.extend(_delete_file_entity(file.id))

    # Get all models
    models = Models.get_all_models()
    log.info(f"Found {len(models)} models to check for knowledge base {id}")

    # Update models that reference this knowledge base
    for model in models:
        if model.meta and hasattr(model.meta, "knowledge"):
            knowledge_list = model.meta.knowledge or []
            # Filter out the deleted knowledge base
            updated_knowledge = [k for k in knowledge_list if k.get("id") != id]

            # If the knowledge list changed, update the model
            if len(updated_knowledge) != len(knowledge_list):
                log.info(f"Updating model {model.id} to remove knowledge base {id}")
                model.meta.knowledge = updated_knowledge
                # Create a ModelForm for the update
                model_form = ModelForm(
                    id=model.id,
                    name=model.name,
                    base_model_id=model.base_model_id,
                    meta=model.meta,
                    params=model.params,
                    access_control=model.access_control,
                    is_active=model.is_active,
                )
                Models.update_model_by_id(model.id, model_form)

    # Clean up vector DB
    active_collection_name = get_active_vector_collection_name(id, knowledge.meta)
    try:
        if VECTOR_DB_CLIENT.has_collection(collection_name=active_collection_name):
            await run_in_threadpool(
                VECTOR_DB_CLIENT.delete_collection,
                collection_name=active_collection_name,
            )
            invalidate_bm25_collections([active_collection_name])
    except Exception as e:
        warnings.append(
            _cleanup_warning(
                "vector.delete_collection",
                active_collection_name,
                e,
            )
        )
    try:
        mark_bm25_collections_dirty([active_collection_name])
    except Exception as e:
        warnings.append(_cleanup_warning("bm25.mark_dirty", active_collection_name, e))

    result = Knowledges.delete_knowledge_by_id(id=id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Failed to delete knowledge base"),
        )

    return DeleteKnowledgeResponse(
        status=True,
        warnings=_format_cleanup_warnings(
            warnings,
            skipped_file_deletions=skipped_file_deletions,
        ),
    )


############################
# ResetKnowledgeById
############################


@router.post("/{id}/reset", response_model=Optional[KnowledgeResponse])
async def reset_knowledge_by_id(id: str, user=Depends(get_verified_user)):
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        knowledge.user_id != user.id
        and not has_access(user.id, "write", knowledge.access_control)
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    try:
        active_collection_name = get_active_vector_collection_name(id, knowledge.meta)
        if VECTOR_DB_CLIENT.has_collection(collection_name=active_collection_name):
            await run_in_threadpool(
                VECTOR_DB_CLIENT.delete_collection,
                collection_name=active_collection_name,
            )
    except Exception as e:
        log.debug(e)
        pass

    knowledge = Knowledges.reset_knowledge_by_id(id=id)
    if knowledge:
        next_meta = upsert_reindex_status_meta(
            knowledge.meta,
            active_collection_name=id,
            status="success",
            success=False,
        )
        knowledge = Knowledges.update_knowledge_by_id(
            id=id,
            form_data=KnowledgeForm(meta=next_meta),
        )
    return knowledge


############################
# AddFilesToKnowledge
############################


@router.post("/{id}/files/batch/add", response_model=Optional[KnowledgeFilesResponse])
async def add_files_to_knowledge_batch(
    request: Request,
    id: str,
    form_data: list[KnowledgeFileIdForm],
    user=Depends(get_verified_user),
):
    """
    Add multiple files to a knowledge base
    """
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        knowledge.user_id != user.id
        and not has_access(user.id, "write", knowledge.access_control)
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    # Get files content
    log.info(f"files/batch/add - {len(form_data)} files")
    files: List[FileModel] = []
    for form in form_data:
        file = Files.get_file_by_id(form.file_id)
        if not file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {form.file_id} not found",
            )
        files.append(file)

    # Process files
    try:
        collection_name = get_active_vector_collection_name(knowledge.id, knowledge.meta)
        result = await process_files_batch(
            request=request,
            form_data=BatchProcessFilesForm(files=files, collection_name=collection_name),
            user=user,
        )
    except Exception as e:
        log.error(
            f"add_files_to_knowledge_batch: Exception occurred: {e}", exc_info=True
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # Only add files that were successfully processed
    successful_file_ids = [r.file_id for r in result.results if r.status == "completed"]
    for file_id in successful_file_ids:
        Knowledges.add_file_to_knowledge_by_id(
            knowledge_id=id, file_id=file_id, user_id=user.id
        )

    # If there were any errors, include them in the response
    if result.errors:
        error_details = [f"{err.file_id}: {err.error}" for err in result.errors]
        return KnowledgeFilesResponse(
            **knowledge.model_dump(),
            files=Knowledges.get_file_metadatas_by_id(knowledge.id),
            warnings={
                "message": "Some files failed to process",
                "errors": error_details,
            },
        )

    return KnowledgeFilesResponse(
        **knowledge.model_dump(),
        files=Knowledges.get_file_metadatas_by_id(knowledge.id),
    )
