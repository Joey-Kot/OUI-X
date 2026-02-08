import time
from typing import Any, Optional

COLLECTION_RAG_OVERRIDE_KEYS = [
    "RAG_EMBEDDING_ENGINE",
    "RAG_EMBEDDING_MODEL",
    "TEXT_SPLITTER",
    "VOYAGE_TOKENIZER_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RAG_EMBEDDING_BATCH_SIZE",
    "RAG_RERANKING_ENGINE",
    "RAG_RERANKING_MODEL",
    "TOP_K",
    "TOP_K_RERANKER",
    "RELEVANCE_THRESHOLD",
]


COLLECTION_RAG_DEFAULTS_KEYS = COLLECTION_RAG_OVERRIDE_KEYS


COLLECTION_RAG_CONFIG_META_KEY = "collection_rag_config"
COLLECTION_VECTOR_META_KEY = "collection_vector"


def get_global_rag_defaults(config: Any) -> dict:
    return {
        "RAG_EMBEDDING_ENGINE": config.RAG_EMBEDDING_ENGINE,
        "RAG_EMBEDDING_MODEL": config.RAG_EMBEDDING_MODEL,
        "TEXT_SPLITTER": config.TEXT_SPLITTER,
        "VOYAGE_TOKENIZER_MODEL": config.VOYAGE_TOKENIZER_MODEL,
        "CHUNK_SIZE": config.CHUNK_SIZE,
        "CHUNK_OVERLAP": config.CHUNK_OVERLAP,
        "RAG_EMBEDDING_BATCH_SIZE": config.RAG_EMBEDDING_BATCH_SIZE,
        "RAG_RERANKING_ENGINE": config.RAG_RERANKING_ENGINE,
        "RAG_RERANKING_MODEL": config.RAG_RERANKING_MODEL,
        "TOP_K": config.TOP_K,
        "TOP_K_RERANKER": config.TOP_K_RERANKER,
        "RELEVANCE_THRESHOLD": config.RELEVANCE_THRESHOLD,
    }


def normalize_collection_rag_config(meta: Optional[dict]) -> dict:
    rag_meta = ((meta or {}).get(COLLECTION_RAG_CONFIG_META_KEY) or {})
    mode = rag_meta.get("mode") if rag_meta.get("mode") in {"default", "custom"} else "default"

    raw_overrides = rag_meta.get("overrides") or {}
    overrides = {
        key: raw_overrides[key]
        for key in COLLECTION_RAG_OVERRIDE_KEYS
        if key in raw_overrides and raw_overrides[key] is not None
    }

    if mode == "default":
        overrides = {}

    return {
        "mode": mode,
        "overrides": overrides,
    }


def resolve_collection_rag_config(meta: Optional[dict], config: Any) -> dict:
    global_defaults = get_global_rag_defaults(config)
    normalized = normalize_collection_rag_config(meta)

    effective = {**global_defaults, **normalized["overrides"]}

    return {
        "mode": normalized["mode"],
        "overrides": normalized["overrides"],
        "global_defaults": global_defaults,
        "effective": effective,
    }


def sanitize_rag_overrides(overrides: Optional[dict]) -> dict:
    sanitized = {}
    for key in COLLECTION_RAG_OVERRIDE_KEYS:
        if key not in (overrides or {}):
            continue

        value = overrides.get(key)
        if value is None:
            continue

        if key in {
            "RAG_EMBEDDING_ENGINE",
            "RAG_EMBEDDING_MODEL",
            "TEXT_SPLITTER",
            "VOYAGE_TOKENIZER_MODEL",
            "RAG_RERANKING_ENGINE",
            "RAG_RERANKING_MODEL",
        }:
            if isinstance(value, str):
                value = value.strip()
            if value != "":
                sanitized[key] = value
        elif key in {"CHUNK_SIZE", "CHUNK_OVERLAP", "RAG_EMBEDDING_BATCH_SIZE", "TOP_K", "TOP_K_RERANKER"}:
            sanitized[key] = int(value)
        elif key == "RELEVANCE_THRESHOLD":
            sanitized[key] = float(value)

    return sanitized


def upsert_collection_rag_meta(meta: Optional[dict], mode: str, overrides: Optional[dict]) -> dict:
    next_meta = dict(meta or {})

    if mode not in {"default", "custom"}:
        mode = "default"

    normalized_overrides = sanitize_rag_overrides(overrides)
    if mode == "default":
        normalized_overrides = {}

    next_meta[COLLECTION_RAG_CONFIG_META_KEY] = {
        "mode": mode,
        "overrides": normalized_overrides,
    }

    return next_meta


def get_active_vector_collection_name(knowledge_id: str, meta: Optional[dict]) -> str:
    vector_meta = ((meta or {}).get(COLLECTION_VECTOR_META_KEY) or {})
    active_name = (vector_meta.get("active_collection_name") or "").strip()
    return active_name or knowledge_id


def upsert_reindex_status_meta(
    meta: Optional[dict],
    *,
    active_collection_name: Optional[str] = None,
    status: Optional[str] = None,
    success: bool = False,
) -> dict:
    next_meta = dict(meta or {})
    vector_meta = dict(next_meta.get(COLLECTION_VECTOR_META_KEY) or {})

    if active_collection_name:
        vector_meta["active_collection_name"] = active_collection_name

    if status:
        vector_meta["last_reindex_status"] = status

    if success:
        vector_meta["reindex_version"] = int(vector_meta.get("reindex_version") or 0) + 1
        vector_meta["last_reindexed_at"] = int(time.time())

    next_meta[COLLECTION_VECTOR_META_KEY] = vector_meta
    return next_meta
