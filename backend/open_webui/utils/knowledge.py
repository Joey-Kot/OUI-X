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
    "ENABLE_RAG_RERANKING",
    "RAG_RERANKING_MODEL",
    "TOP_K",
    "TOP_K_RERANKER",
    "RELEVANCE_THRESHOLD",
    "ENABLE_RAG_BM25_SEARCH",
    "ENABLE_RAG_BM25_ENRICHED_TEXTS",
    "BM25_WEIGHT",
    "RETRIEVAL_CHUNK_EXPANSION",
    "RAG_TEMPLATE",
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
        "ENABLE_RAG_RERANKING": config.ENABLE_RAG_RERANKING,
        "RAG_RERANKING_MODEL": config.RAG_RERANKING_MODEL,
        "TOP_K": config.TOP_K,
        "TOP_K_RERANKER": config.TOP_K_RERANKER,
        "RELEVANCE_THRESHOLD": config.RELEVANCE_THRESHOLD,
        "ENABLE_RAG_BM25_SEARCH": config.ENABLE_RAG_BM25_SEARCH,
        "ENABLE_RAG_BM25_ENRICHED_TEXTS": config.ENABLE_RAG_BM25_ENRICHED_TEXTS,
        "BM25_WEIGHT": config.BM25_WEIGHT,
        "RETRIEVAL_CHUNK_EXPANSION": config.RETRIEVAL_CHUNK_EXPANSION,
        "RAG_TEMPLATE": config.RAG_TEMPLATE,
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

    effective = apply_collection_rag_runtime_guards(
        {**global_defaults, **normalized["overrides"]},
        global_defaults,
    )

    return {
        "mode": normalized["mode"],
        "overrides": normalized["overrides"],
        "global_defaults": global_defaults,
        "effective": effective,
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def apply_collection_rag_runtime_guards(effective: dict, global_defaults: dict) -> dict:
    guarded = dict(effective)

    if not _as_bool(guarded.get("ENABLE_RAG_BM25_SEARCH")):
        guarded["ENABLE_RAG_BM25_ENRICHED_TEXTS"] = False
        guarded["BM25_WEIGHT"] = global_defaults.get("BM25_WEIGHT", 0.5)

    if not _as_bool(guarded.get("ENABLE_RAG_RERANKING")):
        guarded["RAG_RERANKING_ENGINE"] = global_defaults.get(
            "RAG_RERANKING_ENGINE", "external"
        )
        guarded["RAG_RERANKING_MODEL"] = global_defaults.get("RAG_RERANKING_MODEL", "")
        guarded["TOP_K_RERANKER"] = global_defaults.get("TOP_K_RERANKER", 0)
        guarded["RELEVANCE_THRESHOLD"] = global_defaults.get(
            "RELEVANCE_THRESHOLD", 0.0
        )

    return guarded


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
        elif key in {
            "ENABLE_RAG_BM25_SEARCH",
            "ENABLE_RAG_BM25_ENRICHED_TEXTS",
            "ENABLE_RAG_RERANKING",
        }:
            if isinstance(value, bool):
                sanitized[key] = value
            elif isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "on"}:
                    sanitized[key] = True
                elif normalized in {"false", "0", "no", "off"}:
                    sanitized[key] = False
        elif key in {"BM25_WEIGHT"}:
            sanitized[key] = float(value)
        elif key in {"RETRIEVAL_CHUNK_EXPANSION"}:
            sanitized[key] = int(value)
        elif key == "RAG_TEMPLATE":
            if isinstance(value, str):
                if value.strip() != "":
                    sanitized[key] = value
            elif value != "":
                sanitized[key] = value

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
