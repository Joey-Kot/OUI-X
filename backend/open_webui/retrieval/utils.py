import logging
import os
import math
from collections import defaultdict
from typing import Awaitable, Optional, Union

import requests
import aiohttp
import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time
import re
import threading
from dataclasses import dataclass, field

from urllib.parse import quote
from huggingface_hub import snapshot_download
from langchain_classic.retrievers import (
    EnsembleRetriever,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from open_webui.config import VECTOR_DB
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT


from open_webui.models.users import UserModel
from open_webui.models.files import Files
from open_webui.models.knowledge import Knowledges

from open_webui.models.chats import Chats
from open_webui.models.notes import Notes

from open_webui.retrieval.vector.main import GetResult
from open_webui.utils.access_control import has_access
from open_webui.utils.knowledge import (
    get_active_vector_collection_name,
    resolve_collection_rag_config,
)
from open_webui.utils.headers import include_user_info_headers
from open_webui.utils.misc import get_message_list

from open_webui.retrieval.web.utils import get_web_loader
from open_webui.retrieval.loaders.youtube import YoutubeLoader


from open_webui.env import (
    AIOHTTP_CLIENT_TIMEOUT,
    OFFLINE_MODE,
    ENABLE_FORWARD_USER_INFO_HEADERS,
    AIOHTTP_CLIENT_SESSION_SSL,
)
from open_webui.config import (
    RAG_EMBEDDING_QUERY_PREFIX,
    RAG_EMBEDDING_CONTENT_PREFIX,
    RAG_EMBEDDING_PREFIX_FIELD_NAME,
)

log = logging.getLogger(__name__)


from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever


def normalize_doc_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip())


def build_dedup_key(doc: Document) -> str:
    metadata = doc.metadata if isinstance(doc.metadata, dict) else {}

    source_id = None
    for key in ("file_id", "source", "id"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            source_id = str(value).strip()
            break

    start_index = metadata.get("start_index")
    if source_id is not None and start_index is not None:
        return f"meta:{source_id}:{start_index}"

    normalized_text = normalize_doc_text(doc.page_content)
    text_hash = hashlib.sha256(normalized_text.encode()).hexdigest()
    return f"text:{text_hash}"


def dedupe_documents_before_rerank(docs: list[Document]) -> list[Document]:
    deduped_docs: list[Document] = []
    seen_keys: set[str] = set()

    for doc in docs:
        key = build_dedup_key(doc)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_docs.append(doc)

    return deduped_docs


def _get_source_key(metadata: dict) -> Optional[str]:
    for key in ("file_id", "source", "id"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _parse_start_index(metadata: dict) -> Optional[int]:
    start_index = metadata.get("start_index")
    if start_index is None:
        return None

    try:
        return int(start_index)
    except (TypeError, ValueError):
        return None


def _build_source_windows(
    collection_result: Optional[GetResult],
) -> dict[str, list[tuple[int, Document]]]:
    source_windows: dict[str, dict[int, Document]] = defaultdict(dict)
    if not has_collection_documents(collection_result):
        return {}

    documents = collection_result.documents[0]
    metadatas = collection_result.metadatas[0] if collection_result.metadatas else []

    for idx, document in enumerate(documents):
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        if not isinstance(metadata, dict):
            metadata = {}

        source_key = _get_source_key(metadata)
        start_index = _parse_start_index(metadata)
        if source_key is None or start_index is None:
            continue

        source_windows[source_key].setdefault(
            start_index,
            Document(page_content=document, metadata=metadata),
        )

    return {
        source_key: sorted(start_map.items(), key=lambda item: item[0])
        for source_key, start_map in source_windows.items()
    }


def expand_documents_by_window(
    docs: list[Document],
    collection_result: Optional[GetResult],
    window_size: int,
) -> list[Document]:
    if window_size <= 0 or len(docs) == 0:
        return docs

    source_windows = _build_source_windows(collection_result)
    if not source_windows:
        return docs

    source_positions = {
        source_key: {start_index: idx for idx, (start_index, _) in enumerate(entries)}
        for source_key, entries in source_windows.items()
    }

    expanded_docs: list[Document] = []
    for doc in docs:
        expanded_docs.append(doc)

        metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
        inherited_score = metadata.get("score")
        inherited_score = (
            float(inherited_score)
            if isinstance(inherited_score, (int, float))
            else None
        )
        source_key = _get_source_key(metadata)
        start_index = _parse_start_index(metadata)
        if source_key is None or start_index is None:
            continue

        entries = source_windows.get(source_key)
        positions = source_positions.get(source_key)
        if not entries or not positions:
            continue

        center_idx = positions.get(start_index)
        if center_idx is None:
            continue

        for distance in range(1, window_size + 1):
            left_idx = center_idx - distance
            right_idx = center_idx + distance

            if left_idx >= 0:
                left_doc = entries[left_idx][1]
                left_metadata = (
                    dict(left_doc.metadata) if isinstance(left_doc.metadata, dict) else {}
                )
                left_metadata["retrieval_chunk_expanded"] = True
                if inherited_score is not None:
                    left_metadata["score"] = inherited_score
                expanded_docs.append(
                    Document(page_content=left_doc.page_content, metadata=left_metadata)
                )
            if right_idx < len(entries):
                right_doc = entries[right_idx][1]
                right_metadata = (
                    dict(right_doc.metadata)
                    if isinstance(right_doc.metadata, dict)
                    else {}
                )
                right_metadata["retrieval_chunk_expanded"] = True
                if inherited_score is not None:
                    right_metadata["score"] = inherited_score
                expanded_docs.append(
                    Document(page_content=right_doc.page_content, metadata=right_metadata)
                )

    return expanded_docs


def is_youtube_url(url: str) -> bool:
    youtube_regex = r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$"
    return re.match(youtube_regex, url) is not None


def get_loader(request, url: str):
    if is_youtube_url(url):
        return YoutubeLoader(
            url,
            language=request.app.state.config.YOUTUBE_LOADER_LANGUAGE,
            proxy_url=request.app.state.config.YOUTUBE_LOADER_PROXY_URL,
        )
    else:
        return get_web_loader(
            url,
            verify_ssl=request.app.state.config.ENABLE_WEB_LOADER_SSL_VERIFICATION,
            requests_per_second=request.app.state.config.WEB_LOADER_CONCURRENT_REQUESTS,
            trust_env=request.app.state.config.WEB_SEARCH_TRUST_ENV,
        )


def get_content_from_url(request, url: str) -> str:
    loader = get_loader(request, url)
    docs = loader.load()
    content = " ".join([doc.page_content for doc in docs])
    return content, docs


class VectorSearchRetriever(BaseRetriever):
    collection_name: Any
    embedding_function: Any
    top_k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for.
            run_manager: The callback handler to use.

        Returns:
            List of relevant documents.
        """
        return []

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        embedding = await self.embedding_function(query, RAG_EMBEDDING_QUERY_PREFIX)
        result = VECTOR_DB_CLIENT.search(
            collection_name=self.collection_name,
            vectors=[embedding],
            limit=self.top_k,
        )

        ids = result.ids[0]
        metadatas = result.metadatas[0]
        documents = result.documents[0]

        results = []
        for idx in range(len(ids)):
            results.append(
                Document(
                    metadata=metadatas[idx],
                    page_content=documents[idx],
                )
            )
        return results


def query_doc(
    collection_name: str, query_embedding: list[float], k: int, user: UserModel = None
):
    try:
        log.debug(f"query_doc:doc {collection_name}")
        result = VECTOR_DB_CLIENT.search(
            collection_name=collection_name,
            vectors=[query_embedding],
            limit=k,
        )

        if result:
            log.info(f"query_doc:result {result.ids} {result.metadatas}")

        return result
    except Exception as e:
        log.exception(f"Error querying doc {collection_name} with limit {k}: {e}")
        raise e


def get_doc(collection_name: str, user: UserModel = None):
    try:
        log.debug(f"get_doc:doc {collection_name}")
        result = VECTOR_DB_CLIENT.get(collection_name=collection_name)

        if result:
            log.info(f"query_doc:result {result.ids} {result.metadatas}")

        return result
    except Exception as e:
        log.exception(f"Error getting doc {collection_name}: {e}")
        raise e


def get_enriched_texts(collection_result: GetResult) -> list[str]:
    enriched_texts = []
    for idx, text in enumerate(collection_result.documents[0]):
        metadata = collection_result.metadatas[0][idx]
        metadata_parts = [text]

        # Add filename (repeat twice for extra weight in BM25 scoring)
        if metadata.get("name"):
            filename = metadata["name"]
            filename_tokens = (
                filename.replace("_", " ").replace("-", " ").replace(".", " ")
            )
            metadata_parts.append(
                f"Filename: {filename} {filename_tokens} {filename_tokens}"
            )

        # Add title if available
        if metadata.get("title"):
            metadata_parts.append(f"Title: {metadata['title']}")

        # Add document section headings if available (from markdown splitter)
        if metadata.get("headings") and isinstance(metadata["headings"], list):
            headings = " > ".join(str(h) for h in metadata["headings"])
            metadata_parts.append(f"Section: {headings}")

        # Add source URL/path if available
        if metadata.get("source"):
            metadata_parts.append(f"Source: {metadata['source']}")

        # Add snippet for web search results
        if metadata.get("snippet"):
            metadata_parts.append(f"Snippet: {metadata['snippet']}")

        enriched_texts.append(" ".join(metadata_parts))

    return enriched_texts


def has_collection_documents(collection_result: Optional[GetResult]) -> bool:
    return bool(
        collection_result
        and hasattr(collection_result, "documents")
        and hasattr(collection_result, "metadatas")
        and collection_result.documents
        and len(collection_result.documents) > 0
        and collection_result.documents[0]
    )


@dataclass
class BM25CacheEntry:
    retriever: BM25Retriever
    signature: str
    enable_enriched_texts: bool
    dirty: bool = False
    last_build_ts: float = field(default_factory=time.time)
    last_access_ts: float = field(default_factory=time.time)


class BM25IndexManager:
    def __init__(self, refresh_interval_seconds: int = 600, stale_ttl_seconds: int = 3600):
        self.refresh_interval_seconds = refresh_interval_seconds
        self.stale_ttl_seconds = stale_ttl_seconds
        self._entries: dict[str, BM25CacheEntry] = {}
        self._state_lock = threading.Lock()
        self._refresh_task: Optional[asyncio.Task] = None
        self._refreshing: set[str] = set()

    @staticmethod
    def _build_signature(collection_result: GetResult, enable_enriched_texts: bool) -> str:
        ids = collection_result.ids[0] if getattr(collection_result, "ids", None) else []
        docs_len = len(collection_result.documents[0])
        payload = f"{enable_enriched_texts}:{docs_len}:" + "|".join(map(str, ids))
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _build_retriever(
        collection_result: GetResult,
        enable_enriched_texts: bool,
        k: int,
    ) -> BM25Retriever:
        texts = (
            get_enriched_texts(collection_result)
            if enable_enriched_texts
            else collection_result.documents[0]
        )
        retriever = BM25Retriever.from_texts(
            texts=texts,
            metadatas=collection_result.metadatas[0],
        )
        retriever.k = k
        return retriever

    async def get_retriever(
        self,
        collection_name: str,
        collection_result: GetResult,
        *,
        k: int,
        enable_enriched_texts: bool,
    ) -> Optional[BM25Retriever]:
        if not has_collection_documents(collection_result):
            return None

        self._ensure_periodic_refresh_task()
        signature = self._build_signature(collection_result, enable_enriched_texts)
        now = time.time()

        with self._state_lock:
            entry = self._entries.get(collection_name)

        if (
            entry is None
            or entry.signature != signature
            or entry.enable_enriched_texts != enable_enriched_texts
        ):
            retriever = await asyncio.to_thread(
                self._build_retriever,
                collection_result,
                enable_enriched_texts,
                k,
            )
            with self._state_lock:
                self._entries[collection_name] = BM25CacheEntry(
                    retriever=retriever,
                    signature=signature,
                    enable_enriched_texts=enable_enriched_texts,
                    dirty=False,
                    last_build_ts=now,
                    last_access_ts=now,
                )
            self._evict_stale_entries(now)
            return retriever

        entry.last_access_ts = now
        entry.retriever.k = k

        if entry.dirty:
            self._schedule_refresh(collection_name)

        self._evict_stale_entries(now)
        return entry.retriever

    def mark_dirty(self, collection_name: str):
        with self._state_lock:
            entry = self._entries.get(collection_name)
            if entry is not None:
                entry.dirty = True

    def invalidate(self, collection_name: str):
        with self._state_lock:
            self._entries.pop(collection_name, None)
            self._refreshing.discard(collection_name)

    def clear(self):
        with self._state_lock:
            self._entries.clear()
            self._refreshing.clear()

    def _evict_stale_entries(self, now: float):
        if self.stale_ttl_seconds <= 0:
            return

        stale_collections = []
        with self._state_lock:
            for collection_name, entry in self._entries.items():
                if (now - entry.last_access_ts) > self.stale_ttl_seconds:
                    stale_collections.append(collection_name)

            for collection_name in stale_collections:
                self._entries.pop(collection_name, None)
                self._refreshing.discard(collection_name)

    async def _refresh_collection(self, collection_name: str):
        try:
            collection_result = await asyncio.to_thread(
                VECTOR_DB_CLIENT.get,
                collection_name=collection_name,
            )
            if not has_collection_documents(collection_result):
                self.invalidate(collection_name)
                return

            with self._state_lock:
                entry = self._entries.get(collection_name)
            if entry is None:
                return

            signature = self._build_signature(
                collection_result,
                entry.enable_enriched_texts,
            )

            retriever = await asyncio.to_thread(
                self._build_retriever,
                collection_result,
                entry.enable_enriched_texts,
                entry.retriever.k,
            )

            with self._state_lock:
                existing = self._entries.get(collection_name)
                if existing is None:
                    return
                existing.retriever = retriever
                existing.signature = signature
                existing.dirty = False
                existing.last_build_ts = time.time()
                existing.last_access_ts = time.time()
        except Exception:
            log.exception(
                f"Failed to refresh BM25 index cache for collection: {collection_name}"
            )
        finally:
            with self._state_lock:
                self._refreshing.discard(collection_name)

    def _schedule_refresh(self, collection_name: str):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return

        with self._state_lock:
            if collection_name in self._refreshing:
                return
            self._refreshing.add(collection_name)

        asyncio.create_task(self._refresh_collection(collection_name))

    async def _periodic_refresh(self):
        while True:
            await asyncio.sleep(self.refresh_interval_seconds)

            with self._state_lock:
                dirty_collections = [
                    name for name, entry in self._entries.items() if entry.dirty
                ]

            for collection_name in dirty_collections:
                self._schedule_refresh(collection_name)

            self._evict_stale_entries(time.time())

    def _ensure_periodic_refresh_task(self):
        if self._refresh_task is not None and not self._refresh_task.done():
            return

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return

        self._refresh_task = asyncio.create_task(self._periodic_refresh())


BM25_INDEX_MANAGER = BM25IndexManager()


def mark_bm25_collections_dirty(collection_names: list[str]):
    for collection_name in collection_names:
        if collection_name:
            BM25_INDEX_MANAGER.mark_dirty(collection_name)


def invalidate_bm25_collections(collection_names: list[str]):
    for collection_name in collection_names:
        if collection_name:
            BM25_INDEX_MANAGER.invalidate(collection_name)


def clear_bm25_index_cache():
    BM25_INDEX_MANAGER.clear()


async def query_doc_with_rag_pipeline(
    collection_name: str,
    query: str,
    embedding_function,
    k: int,
    reranking_function,
    k_reranker: int,
    r: float,
    bm25_weight: float,
    enable_bm25_search: bool,
    enable_reranking: bool,
    collection_result: Optional[GetResult] = None,
    enable_bm25_enriched_texts: bool = False,
    retrieval_chunk_expansion: int = 0,
) -> dict:
    try:
        vector_search_retriever = VectorSearchRetriever(
            collection_name=collection_name,
            embedding_function=embedding_function,
            top_k=k,
        )

        base_retriever = vector_search_retriever
        if enable_bm25_search and has_collection_documents(collection_result):
            bm25_retriever = await BM25_INDEX_MANAGER.get_retriever(
                collection_name,
                collection_result,
                k=k,
                enable_enriched_texts=enable_bm25_enriched_texts,
            )

            if bm25_retriever is not None:
                if bm25_weight <= 0:
                    base_retriever = EnsembleRetriever(
                        retrievers=[vector_search_retriever],
                        weights=[1.0],
                    )
                elif bm25_weight >= 1:
                    base_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever],
                        weights=[1.0],
                    )
                else:
                    base_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, vector_search_retriever],
                        weights=[bm25_weight, 1.0 - bm25_weight],
                    )

        retrieved_docs = await base_retriever.ainvoke(query)
        retrieved_count = len(retrieved_docs)
        deduped_docs = dedupe_documents_before_rerank(retrieved_docs)
        deduped_count = len(deduped_docs)
        rerank_input_count = deduped_count if enable_reranking else 0

        if enable_reranking:
            compressor = RerankCompressor(
                embedding_function=embedding_function,
                top_n=k_reranker,
                reranking_function=reranking_function,
                r_score=r,
            )
            result_docs = list(
                await compressor.acompress_documents(deduped_docs, query)
            )

            if k < k_reranker:
                result_docs = sorted(
                    result_docs,
                    key=lambda x: x.metadata.get("score", 0.0),
                    reverse=True,
                )[:k]
        else:
            result_docs = deduped_docs
            result_docs = result_docs[:k]

        rerank_output_count = len(result_docs)

        if retrieval_chunk_expansion > 0 and len(result_docs) > 0:
            collection_result_for_expansion = collection_result
            if not has_collection_documents(collection_result_for_expansion):
                collection_result_for_expansion = VECTOR_DB_CLIENT.get(
                    collection_name=collection_name
                )

            result_docs = expand_documents_by_window(
                result_docs,
                collection_result_for_expansion,
                retrieval_chunk_expansion,
            )
            result_docs = dedupe_documents_before_rerank(result_docs)

            final_limit = max(k, k * (2 * retrieval_chunk_expansion + 1))
            if final_limit > 0:
                result_docs = result_docs[:final_limit]

        expanded_output_count = len(result_docs)

        ranked = []
        for idx, doc in enumerate(result_docs):
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            score = metadata.get("score")
            is_expanded = metadata.get("retrieval_chunk_expanded") is True

            if isinstance(score, (int, float)):
                distance = float(score)
            elif is_expanded:
                distance = None
            else:
                distance = float(len(result_docs) - idx)

            ranked.append((distance, doc.page_content, metadata))

        distances = [item[0] for item in ranked]
        documents = [item[1] for item in ranked]
        metadatas = [item[2] for item in ranked]

        result = {
            "distances": [distances],
            "documents": [documents],
            "metadatas": [metadatas],
        }

        log.info(
            "query_doc_with_rag_pipeline:counts "
            + f"retrieved_count={retrieved_count} "
            + f"deduped_count={deduped_count} "
            + f"rerank_input_count={rerank_input_count} "
            + f"rerank_output_count={rerank_output_count} "
            + f"expanded_output_count={expanded_output_count}"
        )
        log.info(
            "query_doc_with_rag_pipeline:result "
            + f'{result["metadatas"]} {result["distances"]}'
        )
        return result
    except Exception as e:
        log.exception(f"Error querying doc {collection_name} with rag pipeline: {e}")
        raise e


async def query_doc_with_hybrid_search(
    collection_name: str,
    collection_result: GetResult,
    query: str,
    embedding_function,
    k: int,
    reranking_function,
    k_reranker: int,
    r: float,
    hybrid_bm25_weight: float,
    enable_enriched_texts: bool = False,
) -> dict:
    return await query_doc_with_rag_pipeline(
        collection_name=collection_name,
        query=query,
        embedding_function=embedding_function,
        k=k,
        reranking_function=reranking_function,
        k_reranker=k_reranker,
        r=r,
        bm25_weight=hybrid_bm25_weight,
        enable_bm25_search=True,
        enable_reranking=True,
        collection_result=collection_result,
        enable_bm25_enriched_texts=enable_enriched_texts,
    )


def merge_get_results(get_results: list[dict]) -> dict:
    # Initialize lists to store combined data
    combined_documents = []
    combined_metadatas = []
    combined_ids = []

    for data in get_results:
        combined_documents.extend(data["documents"][0])
        combined_metadatas.extend(data["metadatas"][0])
        combined_ids.extend(data["ids"][0])

    # Create the output dictionary
    result = {
        "documents": [combined_documents],
        "metadatas": [combined_metadatas],
        "ids": [combined_ids],
    }

    return result


def normalize_merge_distance(distance: Optional[float]) -> float:
    if isinstance(distance, (int, float)):
        return float(distance)
    return float("-inf")


def resolve_expansion_aware_merge_k(
    explicit_k: Optional[int],
    collection_ks: list[int],
    collection_expansions: list[int],
    default_k: int,
) -> int:
    if explicit_k is not None:
        return explicit_k

    if not collection_ks:
        return default_k

    expansion_aware_ks: list[int] = []
    for idx, base_k in enumerate(collection_ks):
        expansion = (
            collection_expansions[idx] if idx < len(collection_expansions) else 0
        )
        normalized_base_k = max(0, int(base_k))
        normalized_expansion = max(0, int(expansion))
        expansion_aware_ks.append(
            normalized_base_k * (2 * normalized_expansion + 1)
        )

    return max(expansion_aware_ks) if expansion_aware_ks else default_k


def merge_and_sort_query_results(query_results: list[dict], k: int) -> dict:
    # Initialize lists to store combined data
    combined = dict()  # To store documents with unique document hashes
    has_none_distance = False

    for data in query_results:
        if (
            len(data.get("distances", [])) == 0
            or len(data.get("documents", [])) == 0
            or len(data.get("metadatas", [])) == 0
        ):
            continue

        distances = data["distances"][0]
        documents = data["documents"][0]
        metadatas = data["metadatas"][0]

        for distance, document, metadata in zip(distances, documents, metadatas):
            normalized_distance = normalize_merge_distance(distance)
            has_none_distance = has_none_distance or distance is None
            if isinstance(document, str):
                doc_hash = hashlib.sha256(
                    document.encode()
                ).hexdigest()  # Compute a hash for uniqueness

                if doc_hash not in combined.keys():
                    combined[doc_hash] = (
                        normalized_distance,
                        distance,
                        document,
                        metadata,
                    )
                    continue  # if doc is new, no further comparison is needed

                # if doc is alredy in, but new distance is better, update
                if normalized_distance > combined[doc_hash][0]:
                    combined[doc_hash] = (
                        normalized_distance,
                        distance,
                        document,
                        metadata,
                    )

    combined = list(combined.values())
    # Sort the list based on distances
    combined.sort(key=lambda x: x[0], reverse=True)
    if combined:
        log.debug(
            "merge_and_sort_query_results:counts "
            + f"input_results={len(query_results)} "
            + f"deduped_count={len(combined)} "
            + f"merge_limit={k} "
            + f"has_none_distance={has_none_distance}"
        )

    # Slice to keep only the top k elements
    sorted_rows = combined[:k] if k > 0 else []
    sorted_distances, sorted_documents, sorted_metadatas = (
        (
            [row[1] for row in sorted_rows],
            [row[2] for row in sorted_rows],
            [row[3] for row in sorted_rows],
        )
        if sorted_rows
        else ([], [], [])
    )

    # Create and return the output dictionary
    return {
        "distances": [sorted_distances],
        "documents": [sorted_documents],
        "metadatas": [sorted_metadatas],
    }


def get_all_items_from_collections(collection_names: list[str]) -> dict:
    results = []

    for collection_name in collection_names:
        if collection_name:
            try:
                result = get_doc(collection_name=collection_name)
                if result is not None:
                    results.append(result.model_dump())
            except Exception as e:
                log.exception(f"Error when querying the collection: {e}")
        else:
            pass

    return merge_get_results(results)


async def query_collection(
    collection_names: list[str],
    queries: list[str],
    embedding_function,
    k: int,
) -> dict:
    results = []
    error = False

    def process_query_collection(collection_name, query_embedding):
        try:
            if collection_name:
                result = query_doc(
                    collection_name=collection_name,
                    k=k,
                    query_embedding=query_embedding,
                )
                if result is not None:
                    return result.model_dump(), None
            return None, None
        except Exception as e:
            log.exception(f"Error when querying the collection: {e}")
            return None, e

    # Generate all query embeddings (in one call)
    query_embeddings = await embedding_function(
        queries, prefix=RAG_EMBEDDING_QUERY_PREFIX
    )
    log.debug(
        f"query_collection: processing {len(queries)} queries across {len(collection_names)} collections"
    )

    with ThreadPoolExecutor() as executor:
        future_results = []
        for query_embedding in query_embeddings:
            for collection_name in collection_names:
                result = executor.submit(
                    process_query_collection, collection_name, query_embedding
                )
                future_results.append(result)
        task_results = [future.result() for future in future_results]

    for result, err in task_results:
        if err is not None:
            error = True
        elif result is not None:
            results.append(result)

    if error and not results:
        log.warning("All collection queries failed. No results returned.")

    return merge_and_sort_query_results(results, k=k)


async def query_collection_with_rag_pipeline(
    collection_names: list[str],
    queries: list[str],
    embedding_function,
    k: int,
    reranking_function,
    k_reranker: int,
    r: float,
    bm25_weight: float,
    enable_bm25_search: bool,
    enable_reranking: bool,
    enable_bm25_enriched_texts: bool = False,
) -> dict:
    results = []
    error = False

    collection_results = {}
    if enable_bm25_search:
        for collection_name in collection_names:
            try:
                log.debug(
                    "query_collection_with_rag_pipeline:VECTOR_DB_CLIENT.get:collection "
                    f"{collection_name}"
                )
                collection_results[collection_name] = VECTOR_DB_CLIENT.get(
                    collection_name=collection_name
                )
            except Exception as e:
                log.exception(f"Failed to fetch collection {collection_name}: {e}")
                collection_results[collection_name] = None

    log.info(
        "Starting retrieval pipeline for "
        f"{len(queries)} queries in {len(collection_names)} collections..."
    )

    async def process_query(collection_name: str, query: str):
        try:
            result = await query_doc_with_rag_pipeline(
                collection_name=collection_name,
                collection_result=collection_results.get(collection_name),
                query=query,
                embedding_function=embedding_function,
                k=k,
                reranking_function=reranking_function,
                k_reranker=k_reranker,
                r=r,
                bm25_weight=bm25_weight,
                enable_bm25_search=enable_bm25_search,
                enable_reranking=enable_reranking,
                enable_bm25_enriched_texts=enable_bm25_enriched_texts,
            )
            return result, None
        except Exception as e:
            log.exception(
                f"Error when querying the collection with retrieval pipeline: {e}"
            )
            return None, e

    tasks = [
        (collection_name, query)
        for collection_name in collection_names
        if (not enable_bm25_search) or collection_results.get(collection_name) is not None
        for query in queries
    ]

    task_results = await asyncio.gather(
        *[process_query(collection_name, query) for collection_name, query in tasks]
    )

    for result, err in task_results:
        if err is not None:
            error = True
        elif result is not None:
            results.append(result)

    if error and not results:
        raise Exception(
            "Retrieval pipeline failed for all collections. Using vector search as fallback."
        )

    return merge_and_sort_query_results(results, k=k)


async def query_collection_with_hybrid_search(
    collection_names: list[str],
    queries: list[str],
    embedding_function,
    k: int,
    reranking_function,
    k_reranker: int,
    r: float,
    hybrid_bm25_weight: float,
    enable_enriched_texts: bool = False,
) -> dict:
    return await query_collection_with_rag_pipeline(
        collection_names=collection_names,
        queries=queries,
        embedding_function=embedding_function,
        k=k,
        reranking_function=reranking_function,
        k_reranker=k_reranker,
        r=r,
        bm25_weight=hybrid_bm25_weight,
        enable_bm25_search=True,
        enable_reranking=True,
        enable_bm25_enriched_texts=enable_enriched_texts,
    )


async def query_doc_with_file_scope(
    collection_name: str,
    file_id: str,
    query: str,
    embedding_function,
    k: int,
    reranking_function,
    k_reranker: int,
    r: float,
    enable_reranking: bool,
    retrieval_chunk_expansion: int = 0,
) -> Optional[dict]:
    try:
        scoped_result = VECTOR_DB_CLIENT.query(
            collection_name=collection_name,
            filter={"file_id": file_id},
        )
        if not has_collection_documents(scoped_result):
            return None

        scoped_docs = [
            Document(
                page_content=document,
                metadata=(
                    scoped_result.metadatas[0][idx]
                    if scoped_result.metadatas and scoped_result.metadatas[0]
                    else {}
                ),
            )
            for idx, document in enumerate(scoped_result.documents[0])
        ]
        if len(scoped_docs) == 0:
            return None

        base_retriever = BM25Retriever.from_documents(scoped_docs)
        base_retriever.k = max(k, k_reranker)

        retrieved_docs = await base_retriever.ainvoke(query)
        retrieved_count = len(retrieved_docs)
        deduped_docs = dedupe_documents_before_rerank(retrieved_docs)
        deduped_count = len(deduped_docs)
        rerank_input_count = deduped_count if enable_reranking else 0

        if enable_reranking:
            compressor = RerankCompressor(
                embedding_function=embedding_function,
                top_n=k_reranker,
                reranking_function=reranking_function,
                r_score=r,
            )
            result_docs = list(
                await compressor.acompress_documents(deduped_docs, query)
            )
            if k < k_reranker:
                result_docs = sorted(
                    result_docs,
                    key=lambda x: x.metadata.get("score", 0.0),
                    reverse=True,
                )[:k]
        else:
            result_docs = deduped_docs
            result_docs = result_docs[:k]

        rerank_output_count = len(result_docs)

        if retrieval_chunk_expansion > 0 and len(result_docs) > 0:
            result_docs = expand_documents_by_window(
                result_docs,
                scoped_result,
                retrieval_chunk_expansion,
            )
            result_docs = dedupe_documents_before_rerank(result_docs)

            final_limit = max(k, k * (2 * retrieval_chunk_expansion + 1))
            if final_limit > 0:
                result_docs = result_docs[:final_limit]

        expanded_output_count = len(result_docs)

        ranked = []
        for idx, doc in enumerate(result_docs):
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            score = metadata.get("score")
            is_expanded = metadata.get("retrieval_chunk_expanded") is True

            if isinstance(score, (int, float)):
                distance = float(score)
            elif is_expanded:
                distance = None
            else:
                distance = float(len(result_docs) - idx)

            ranked.append((distance, doc.page_content, metadata))

        result = {
            "distances": [[item[0] for item in ranked]],
            "documents": [[item[1] for item in ranked]],
            "metadatas": [[item[2] for item in ranked]],
        }
        log.info(
            "query_doc_with_file_scope:counts "
            + f"retrieved_count={retrieved_count} "
            + f"deduped_count={deduped_count} "
            + f"rerank_input_count={rerank_input_count} "
            + f"rerank_output_count={rerank_output_count} "
            + f"expanded_output_count={expanded_output_count}"
        )
        return result
    except Exception as e:
        log.exception(
            f"Error querying file-scoped docs from {collection_name} with file_id={file_id}: {e}"
        )
        return None


def generate_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str = "https://api.openai.com/v1",
    key: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"generate_openai_batch_embeddings:model {model} batch size: {len(texts)}"
        )
        json_data = {"input": texts, "model": model}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        if ENABLE_FORWARD_USER_INFO_HEADERS and user:
            headers = include_user_info_headers(headers, user)

        r = requests.post(
            f"{url}/embeddings",
            headers=headers,
            json=json_data,
        )
        r.raise_for_status()
        data = r.json()
        if "data" in data:
            return [elem["embedding"] for elem in data["data"]]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        log.exception(f"Error generating openai batch embeddings: {e}")
        return None


async def agenerate_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str = "https://api.openai.com/v1",
    key: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"agenerate_openai_batch_embeddings:model {model} batch size: {len(texts)}"
        )
        form_data = {"input": texts, "model": model}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            form_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        if ENABLE_FORWARD_USER_INFO_HEADERS and user:
            headers = include_user_info_headers(headers, user)

        async with aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
        ) as session:
            async with session.post(
                f"{url}/embeddings", headers=headers, json=form_data
            ) as r:
                r.raise_for_status()
                data = await r.json()
                if "data" in data:
                    return [item["embedding"] for item in data["data"]]
                else:
                    raise Exception("Something went wrong :/")
    except Exception as e:
        log.exception(f"Error generating openai batch embeddings: {e}")
        return None


def generate_azure_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str,
    key: str = "",
    version: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"generate_azure_openai_batch_embeddings:deployment {model} batch size: {len(texts)}"
        )
        json_data = {"input": texts}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        url = f"{url}/openai/deployments/{model}/embeddings?api-version={version}"

        for _ in range(5):
            headers = {
                "Content-Type": "application/json",
                "api-key": key,
            }
            if ENABLE_FORWARD_USER_INFO_HEADERS and user:
                headers = include_user_info_headers(headers, user)

            r = requests.post(
                url,
                headers=headers,
                json=json_data,
            )
            if r.status_code == 429:
                retry = float(r.headers.get("Retry-After", "1"))
                time.sleep(retry)
                continue
            r.raise_for_status()
            data = r.json()
            if "data" in data:
                return [elem["embedding"] for elem in data["data"]]
            else:
                raise Exception("Something went wrong :/")
        return None
    except Exception as e:
        log.exception(f"Error generating azure openai batch embeddings: {e}")
        return None


async def agenerate_azure_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str,
    key: str = "",
    version: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"agenerate_azure_openai_batch_embeddings:deployment {model} batch size: {len(texts)}"
        )
        form_data = {"input": texts}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            form_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        full_url = f"{url}/openai/deployments/{model}/embeddings?api-version={version}"

        headers = {
            "Content-Type": "application/json",
            "api-key": key,
        }
        if ENABLE_FORWARD_USER_INFO_HEADERS and user:
            headers = include_user_info_headers(headers, user)

        async with aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
        ) as session:
            async with session.post(full_url, headers=headers, json=form_data) as r:
                r.raise_for_status()
                data = await r.json()
                if "data" in data:
                    return [item["embedding"] for item in data["data"]]
                else:
                    raise Exception("Something went wrong :/")
    except Exception as e:
        log.exception(f"Error generating azure openai batch embeddings: {e}")
        return None


def get_embedding_function(
    embedding_engine,
    embedding_model,
    embedding_function,
    url,
    key,
    embedding_batch_size,
    azure_api_version=None,
    enable_async=True,
) -> Awaitable:
    if embedding_engine in ["openai", "azure_openai"]:
        embedding_function = lambda query, prefix=None, user=None: generate_embeddings(
            engine=embedding_engine,
            model=embedding_model,
            text=query,
            prefix=prefix,
            url=url,
            key=key,
            user=user,
            azure_api_version=azure_api_version,
        )

        async def async_embedding_function(query, prefix=None, user=None):
            if isinstance(query, list):
                # Create batches
                batches = [
                    query[i : i + embedding_batch_size]
                    for i in range(0, len(query), embedding_batch_size)
                ]

                if enable_async:
                    log.debug(
                        f"generate_multiple_async: Processing {len(batches)} batches in parallel"
                    )
                    # Execute all batches in parallel
                    tasks = [
                        embedding_function(batch, prefix=prefix, user=user)
                        for batch in batches
                    ]
                    batch_results = await asyncio.gather(*tasks)
                else:
                    log.debug(
                        f"generate_multiple_async: Processing {len(batches)} batches sequentially"
                    )
                    batch_results = []
                    for batch in batches:
                        batch_results.append(
                            await embedding_function(batch, prefix=prefix, user=user)
                        )

                # Flatten results
                embeddings = []
                for batch_embeddings in batch_results:
                    if isinstance(batch_embeddings, list):
                        embeddings.extend(batch_embeddings)

                log.debug(
                    f"generate_multiple_async: Generated {len(embeddings)} embeddings from {len(batches)} parallel batches"
                )
                return embeddings
            else:
                return await embedding_function(query, prefix, user)

        return async_embedding_function
    else:
        raise ValueError(f"Unknown embedding engine: {embedding_engine}")


async def generate_embeddings(
    engine: str,
    model: str,
    text: Union[str, list[str]],
    prefix: Union[str, None] = None,
    **kwargs,
):
    url = kwargs.get("url", "")
    key = kwargs.get("key", "")
    user = kwargs.get("user")

    if prefix is not None and RAG_EMBEDDING_PREFIX_FIELD_NAME is None:
        if isinstance(text, list):
            text = [f"{prefix}{text_element}" for text_element in text]
        else:
            text = f"{prefix}{text}"

    if engine == "openai":
        embeddings = await agenerate_openai_batch_embeddings(
            model, text if isinstance(text, list) else [text], url, key, prefix, user
        )
        return embeddings[0] if isinstance(text, str) else embeddings
    elif engine == "azure_openai":
        azure_api_version = kwargs.get("azure_api_version", "")
        embeddings = await agenerate_azure_openai_batch_embeddings(
            model,
            text if isinstance(text, list) else [text],
            url,
            key,
            azure_api_version,
            prefix,
            user,
        )
        return embeddings[0] if isinstance(text, str) else embeddings


def get_reranking_function(reranking_engine, reranking_model, reranking_function):
    if reranking_function is None:
        return None
    if reranking_engine in {"external", "voyage"}:
        return lambda query, documents, user=None: reranking_function.predict(
            [(query, doc.page_content) for doc in documents], user=user
        )
    else:
        return lambda query, documents, user=None: reranking_function.predict(
            [(query, doc.page_content) for doc in documents]
        )


def _build_collection_runtime_functions(request, collection_name: str):
    knowledge = Knowledges.get_knowledge_by_id(collection_name)
    if not knowledge:
        return {
            "physical_collection_name": collection_name,
            "effective_config": resolve_collection_rag_config(None, request.app.state.config)[
                "effective"
            ],
            "embedding_function": request.app.state.EMBEDDING_FUNCTION,
            "reranking_function": request.app.state.RERANKING_FUNCTION,
        }

    effective_config = resolve_collection_rag_config(
        knowledge.meta, request.app.state.config
    )["effective"]

    embedding_function = get_embedding_function(
        effective_config["RAG_EMBEDDING_ENGINE"],
        effective_config["RAG_EMBEDDING_MODEL"],
        request.app.state.ef,
        (
            request.app.state.config.RAG_OPENAI_API_BASE_URL
            if effective_config["RAG_EMBEDDING_ENGINE"] == "openai"
            else request.app.state.config.RAG_AZURE_OPENAI_BASE_URL
        ),
        (
            request.app.state.config.RAG_OPENAI_API_KEY
            if effective_config["RAG_EMBEDDING_ENGINE"] == "openai"
            else request.app.state.config.RAG_AZURE_OPENAI_API_KEY
        ),
        effective_config["RAG_EMBEDDING_BATCH_SIZE"],
        azure_api_version=(
            request.app.state.config.RAG_AZURE_OPENAI_API_VERSION
            if effective_config["RAG_EMBEDDING_ENGINE"] == "azure_openai"
            else None
        ),
        enable_async=request.app.state.config.ENABLE_ASYNC_EMBEDDING,
    )

    reranking_function = None
    model = effective_config.get("RAG_RERANKING_MODEL", "")
    if model:
        try:
            if effective_config.get("RAG_RERANKING_ENGINE") == "voyage":
                from open_webui.retrieval.models.voyage import VoyageReranker

                rf = VoyageReranker(
                    url=request.app.state.config.RAG_VOYAGE_RERANKER_URL,
                    api_key=request.app.state.config.RAG_VOYAGE_RERANKER_API_KEY,
                    model=model,
                    timeout=(
                        int(request.app.state.config.RAG_VOYAGE_RERANKER_TIMEOUT)
                        if request.app.state.config.RAG_VOYAGE_RERANKER_TIMEOUT
                        else None
                    ),
                )
            else:
                from open_webui.retrieval.models.external import ExternalReranker

                rf = ExternalReranker(
                    url=request.app.state.config.RAG_EXTERNAL_RERANKER_URL,
                    api_key=request.app.state.config.RAG_EXTERNAL_RERANKER_API_KEY,
                    model=model,
                    timeout=(
                        int(request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT)
                        if request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT
                        else None
                    ),
                )
            reranking_function = get_reranking_function(
                effective_config.get("RAG_RERANKING_ENGINE", "external"),
                model,
                rf,
            )
        except Exception as e:
            log.warning(f"Failed to build collection reranker for {collection_name}: {e}")

    return {
        "physical_collection_name": get_active_vector_collection_name(
            knowledge.id, knowledge.meta
        ),
        "effective_config": effective_config,
        "embedding_function": embedding_function,
        "reranking_function": reranking_function,
    }


async def get_sources_from_items(
    request,
    items,
    queries,
    embedding_function,
    k: Optional[int],
    reranking_function,
    k_reranker,
    r,
    bm25_weight,
    enable_bm25_search,
    enable_reranking,
    enable_bm25_enriched_texts,
    retrieval_chunk_expansion=0,
    full_context=False,
    user: Optional[UserModel] = None,
):
    log.debug(
        f"items: {items} {queries} {embedding_function} {reranking_function} {full_context}"
    )

    extracted_collections = []
    query_results = []

    for item in items:
        query_result = None
        collection_names = []

        if item.get("type") == "text":
            # Raw Text
            # Used during temporary chat file uploads or web page & youtube attachements

            if item.get("context") == "full":
                if item.get("file"):
                    # if item has file data, use it
                    query_result = {
                        "documents": [
                            [item.get("file", {}).get("data", {}).get("content")]
                        ],
                        "metadatas": [[item.get("file", {}).get("meta", {})]],
                    }

            if query_result is None:
                # Fallback
                if item.get("collection_name"):
                    # If item has a collection name, use it
                    collection_names.append(item.get("collection_name"))
                elif item.get("file"):
                    # If item has file data, use it
                    query_result = {
                        "documents": [
                            [item.get("file", {}).get("data", {}).get("content")]
                        ],
                        "metadatas": [[item.get("file", {}).get("meta", {})]],
                    }
                else:
                    # Fallback to item content
                    query_result = {
                        "documents": [[item.get("content")]],
                        "metadatas": [
                            [{"file_id": item.get("id"), "name": item.get("name")}]
                        ],
                    }

        elif item.get("type") == "note":
            # Note Attached
            note = Notes.get_note_by_id(item.get("id"))

            if note and (
                user.role == "admin"
                or note.user_id == user.id
                or has_access(user.id, "read", note.access_control)
            ):
                # User has access to the note
                query_result = {
                    "documents": [[note.data.get("content", {}).get("md", "")]],
                    "metadatas": [[{"file_id": note.id, "name": note.title}]],
                }

        elif item.get("type") == "chat":
            # Chat Attached
            chat = Chats.get_chat_by_id(item.get("id"))

            if chat and (user.role == "admin" or chat.user_id == user.id):
                messages_map = chat.chat.get("history", {}).get("messages", {})
                message_id = chat.chat.get("history", {}).get("currentId")

                if messages_map and message_id:
                    # Reconstruct the message list in order
                    message_list = get_message_list(messages_map, message_id)
                    message_history = "\n".join(
                        [
                            f"#### {m.get('role', 'user').capitalize()}\n{m.get('content')}\n"
                            for m in message_list
                        ]
                    )

                    # User has access to the chat
                    query_result = {
                        "documents": [[message_history]],
                        "metadatas": [[{"file_id": chat.id, "name": chat.title}]],
                    }

        elif item.get("type") == "url":
            content, docs = get_content_from_url(request, item.get("url"))
            if docs:
                query_result = {
                    "documents": [[content]],
                    "metadatas": [[{"url": item.get("url"), "name": item.get("url")}]],
                }
        elif item.get("type") == "file":
            if (
                item.get("context") == "full"
                or request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL
            ):
                if item.get("file", {}).get("data", {}).get("content", ""):
                    # Manual Full Mode Toggle
                    # Used from chat file modal, we can assume that the file content will be available from item.get("file").get("data", {}).get("content")
                    query_result = {
                        "documents": [
                            [item.get("file", {}).get("data", {}).get("content", "")]
                        ],
                        "metadatas": [
                            [
                                {
                                    "file_id": item.get("id"),
                                    "name": item.get("name"),
                                    **item.get("file")
                                    .get("data", {})
                                    .get("metadata", {}),
                                }
                            ]
                        ],
                    }
                elif item.get("id"):
                    file_object = Files.get_file_by_id(item.get("id"))
                    if file_object:
                        query_result = {
                            "documents": [[file_object.data.get("content", "")]],
                            "metadatas": [
                                [
                                    {
                                        "file_id": item.get("id"),
                                        "name": file_object.filename,
                                        "source": file_object.filename,
                                    }
                                ]
                            ],
                        }
            else:
                item_meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
                file_meta = (
                    item.get("file", {}).get("meta", {})
                    if isinstance(item.get("file"), dict)
                    else {}
                )
                item_id = item.get("id")
                file_object = Files.get_file_by_id(item_id) if item_id else None
                file_db_meta = (file_object.meta or {}) if file_object else {}

                conversation_upload_knowledge_id = (
                    item.get("conversation_upload_knowledge_id")
                    or item_meta.get("conversation_upload_knowledge_id")
                    or file_meta.get("conversation_upload_knowledge_id")
                    or file_db_meta.get("conversation_upload_knowledge_id")
                    or item.get("collection_name")
                    or item_meta.get("collection_name")
                    or file_meta.get("collection_name")
                    or file_db_meta.get("collection_name")
                )
                active_collection_name = (
                    item.get("active_collection_name")
                    or item_meta.get("active_collection_name")
                    or file_meta.get("active_collection_name")
                    or file_db_meta.get("active_collection_name")
                )

                if (conversation_upload_knowledge_id or active_collection_name) and item_id:
                    runtime = _build_collection_runtime_functions(
                        request,
                        conversation_upload_knowledge_id or active_collection_name,
                    )
                    effective_config = runtime["effective_config"]
                    collection_name = (
                        active_collection_name or runtime["physical_collection_name"]
                    )
                    collection_embedding_function = runtime["embedding_function"]
                    collection_reranking_function = runtime.get("reranking_function")
                    collection_enable_reranking = effective_config[
                        "ENABLE_RAG_RERANKING"
                    ]

                    scoped_results = []
                    for query in queries:
                        scoped_result = await query_doc_with_file_scope(
                            collection_name=collection_name,
                            file_id=item_id,
                            query=query,
                            embedding_function=lambda q, prefix, ef=collection_embedding_function: ef(
                                q,
                                prefix=prefix,
                                user=user,
                            ),
                            k=effective_config["TOP_K"],
                            reranking_function=(
                                (
                                    lambda q, documents, rf=collection_reranking_function: rf(
                                        q,
                                        documents,
                                        user=user,
                                    )
                                )
                                if collection_reranking_function
                                and collection_enable_reranking
                                else None
                            ),
                            k_reranker=effective_config["TOP_K_RERANKER"],
                            r=effective_config["RELEVANCE_THRESHOLD"],
                            enable_reranking=collection_enable_reranking,
                            retrieval_chunk_expansion=effective_config[
                                "RETRIEVAL_CHUNK_EXPANSION"
                            ],
                        )
                        if scoped_result is not None:
                            scoped_results.append(scoped_result)

                    if scoped_results:
                        scoped_merge_k = resolve_expansion_aware_merge_k(
                            explicit_k=None,
                            collection_ks=[effective_config["TOP_K"]],
                            collection_expansions=[
                                effective_config["RETRIEVAL_CHUNK_EXPANSION"]
                            ],
                            default_k=effective_config["TOP_K"],
                        )
                        log.debug(
                            "get_sources_from_items:file_scope_merge_k "
                            + f"top_k={effective_config['TOP_K']} "
                            + "explicit_k_passed=False "
                            + f"retrieval_chunk_expansion={effective_config['RETRIEVAL_CHUNK_EXPANSION']} "
                            + f"final_merge_k={scoped_merge_k}"
                        )
                        query_result = merge_and_sort_query_results(
                            scoped_results,
                            k=scoped_merge_k,
                        )
                elif not request.app.state.config.CONVERSATION_FILE_UPLOAD_EMBEDDING:
                    # When disabled, chat uploads only participate via full-context mode.
                    continue
                else:
                    # Legacy fallback to per-file collection names.
                    if item.get("legacy"):
                        collection_names.append(f"{item['id']}")
                    else:
                        collection_names.append(f"file-{item['id']}")

        elif item.get("type") == "collection":
            # Manual Full Mode Toggle for Collection
            knowledge_base = Knowledges.get_knowledge_by_id(item.get("id"))

            if knowledge_base and (
                user.role == "admin"
                or knowledge_base.user_id == user.id
                or has_access(user.id, "read", knowledge_base.access_control)
            ):
                if (
                    item.get("context") == "full"
                    or request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL
                ):
                    if knowledge_base and (
                        user.role == "admin"
                        or knowledge_base.user_id == user.id
                        or has_access(user.id, "read", knowledge_base.access_control)
                    ):
                        files = Knowledges.get_files_by_id(knowledge_base.id)

                        documents = []
                        metadatas = []
                        for file in files:
                            documents.append(file.data.get("content", ""))
                            metadatas.append(
                                {
                                    "file_id": file.id,
                                    "name": file.filename,
                                    "source": file.filename,
                                }
                            )

                        query_result = {
                            "documents": [documents],
                            "metadatas": [metadatas],
                        }
                else:
                    # Fallback to collection names
                    if item.get("legacy"):
                        collection_names = item.get("collection_names", [])
                    else:
                        collection_names.append(item["id"])

        elif item.get("docs"):
            # BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL
            query_result = {
                "documents": [[doc.get("content") for doc in item.get("docs")]],
                "metadatas": [[doc.get("metadata") for doc in item.get("docs")]],
            }
        elif item.get("collection_name"):
            # Direct Collection Name
            collection_names.append(item["collection_name"])
        elif item.get("collection_names"):
            # Collection Names List
            collection_names.extend(item["collection_names"])

        # If query_result is None
        # Fallback to collection names and vector search the collections
        if query_result is None and collection_names:
            collection_names = set(collection_names).difference(extracted_collections)
            if not collection_names:
                log.debug(f"skipping {item} as it has already been extracted")
                continue

            try:
                if full_context:
                    query_result = get_all_items_from_collections(collection_names)
                else:
                    query_result = None
                    merge_k = k if k is not None else request.app.state.config.TOP_K
                    try:
                        per_collection_results = []
                        resolved_collection_ks = []
                        resolved_collection_expansions = []
                        for collection_name in collection_names:
                            runtime = _build_collection_runtime_functions(
                                request, collection_name
                            )
                            effective_config = runtime["effective_config"]
                            collection_k = (
                                k if k is not None else effective_config["TOP_K"]
                            )
                            resolved_collection_ks.append(collection_k)
                            resolved_collection_expansions.append(
                                effective_config["RETRIEVAL_CHUNK_EXPANSION"]
                            )
                            physical_collection_name = runtime[
                                "physical_collection_name"
                            ]
                            collection_embedding_function = runtime[
                                "embedding_function"
                            ]
                            collection_reranking_function = runtime.get(
                                "reranking_function"
                            )
                            collection_enable_reranking = effective_config[
                                "ENABLE_RAG_RERANKING"
                            ]

                            collection_result = None
                            collection_enable_bm25_search = effective_config[
                                "ENABLE_RAG_BM25_SEARCH"
                            ]
                            if collection_enable_bm25_search:
                                collection_result = VECTOR_DB_CLIENT.get(
                                    collection_name=physical_collection_name
                                )

                            for query in queries:
                                result = await query_doc_with_rag_pipeline(
                                    collection_name=physical_collection_name,
                                    collection_result=collection_result,
                                    query=query,
                                    embedding_function=lambda q, prefix, ef=collection_embedding_function: ef(
                                        q,
                                        prefix=prefix,
                                        user=user,
                                    ),
                                    k=collection_k,
                                    reranking_function=(
                                        (
                                            lambda q, documents, rf=collection_reranking_function: rf(
                                                q,
                                                documents,
                                                user=user,
                                            )
                                        )
                                        if collection_reranking_function
                                        and collection_enable_reranking
                                        else None
                                    ),
                                    k_reranker=effective_config["TOP_K_RERANKER"],
                                    r=effective_config["RELEVANCE_THRESHOLD"],
                                    bm25_weight=effective_config["BM25_WEIGHT"],
                                    enable_bm25_search=collection_enable_bm25_search,
                                    enable_reranking=collection_enable_reranking,
                                    enable_bm25_enriched_texts=effective_config[
                                        "ENABLE_RAG_BM25_ENRICHED_TEXTS"
                                    ],
                                    retrieval_chunk_expansion=effective_config[
                                        "RETRIEVAL_CHUNK_EXPANSION"
                                    ],
                                )
                                per_collection_results.append(result)

                        merge_k = resolve_expansion_aware_merge_k(
                            explicit_k=k,
                            collection_ks=resolved_collection_ks,
                            collection_expansions=resolved_collection_expansions,
                            default_k=request.app.state.config.TOP_K,
                        )
                        log.debug(
                            "get_sources_from_items:resolved_top_k "
                            + f"explicit_k={k} "
                            + f"explicit_k_passed={k is not None} "
                            + f"collection_ks={resolved_collection_ks} "
                            + f"collection_expansions={resolved_collection_expansions} "
                            + f"final_merge_k={merge_k}"
                        )
                        query_result = merge_and_sort_query_results(
                            per_collection_results, k=merge_k
                        )
                    except Exception:
                        log.debug(
                            "Error when using retrieval pipeline, using vector search as fallback."
                        )

                    if query_result is None:
                        physical_collection_names = [
                            _build_collection_runtime_functions(request, name)[
                                "physical_collection_name"
                            ]
                            for name in collection_names
                        ]
                        query_result = await query_collection(
                            collection_names=physical_collection_names,
                            queries=queries,
                            embedding_function=embedding_function,
                            k=merge_k,
                        )
            except Exception as e:
                log.exception(e)

            extracted_collections.extend(collection_names)

        if query_result:
            if "data" in item:
                del item["data"]
            query_results.append({**query_result, "file": item})

    sources = []
    for query_result in query_results:
        try:
            if "documents" in query_result:
                if "metadatas" in query_result:
                    source = {
                        "source": query_result["file"],
                        "document": query_result["documents"][0],
                        "metadata": query_result["metadatas"][0],
                    }
                    if "distances" in query_result and query_result["distances"]:
                        source["distances"] = query_result["distances"][0]

                    sources.append(source)
        except Exception as e:
            log.exception(e)
    return sources


def get_model_path(model: str, update_model: bool = False):
    # Construct huggingface_hub kwargs with local_files_only to return the snapshot path
    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")

    local_files_only = not update_model

    if OFFLINE_MODE:
        local_files_only = True

    snapshot_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }

    log.debug(f"model: {model}")
    log.debug(f"snapshot_kwargs: {snapshot_kwargs}")

    # Inspiration from upstream sentence_transformers
    if (
        os.path.exists(model)
        or ("\\" in model or model.count("/") > 1)
        and local_files_only
    ):
        # If fully qualified path exists, return input, else set repo_id
        return model
    elif "/" not in model:
        # Set valid repo_id for model short-name
        model = "sentence-transformers" + "/" + model

    snapshot_kwargs["repo_id"] = model

    # Attempt to query the huggingface_hub library to determine the local path and/or to update
    try:
        model_repo_path = snapshot_download(**snapshot_kwargs)
        log.debug(f"model_repo_path: {model_repo_path}")
        return model_repo_path
    except Exception as e:
        log.exception(f"Cannot determine model snapshot path: {e}")
        return model


import operator
from typing import Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document


class RerankCompressor(BaseDocumentCompressor):
    embedding_function: Any
    top_n: int
    reranking_function: Any
    r_score: float

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    @staticmethod
    def _as_vector_list(value: Any) -> list[float]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        return [float(item) for item in value]

    @staticmethod
    def _cosine_similarity_scores(
        query_embedding: Any, document_embeddings: Any
    ) -> list[float]:
        query_vector = RerankCompressor._as_vector_list(query_embedding)
        query_norm = math.sqrt(sum(component * component for component in query_vector))

        if query_norm == 0.0:
            return [0.0 for _ in document_embeddings]

        scores: list[float] = []
        for document_embedding in document_embeddings:
            doc_vector = RerankCompressor._as_vector_list(document_embedding)
            doc_norm = math.sqrt(sum(component * component for component in doc_vector))

            if doc_norm == 0.0:
                scores.append(0.0)
                continue

            dot = sum(a * b for a, b in zip(query_vector, doc_vector))
            scores.append(dot / (query_norm * doc_norm))

        return scores

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context.

        Args:
            documents: The retrieved documents.
            query: The query context.
            callbacks: Optional callbacks to run during compression.

        Returns:
            The compressed documents.

        """
        return []

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        reranking = self.reranking_function is not None

        scores = None
        if reranking:
            scores = await asyncio.to_thread(self.reranking_function, query, documents)
        else:
            query_embedding = await self.embedding_function(
                query, RAG_EMBEDDING_QUERY_PREFIX
            )
            document_embedding = await self.embedding_function(
                [doc.page_content for doc in documents], RAG_EMBEDDING_CONTENT_PREFIX
            )
            scores = self._cosine_similarity_scores(query_embedding, document_embedding)

        if scores is not None:
            docs_with_scores = list(
                zip(
                    documents,
                    scores.tolist() if not isinstance(scores, list) else scores,
                )
            )
            if self.r_score:
                docs_with_scores = [
                    (d, s) for d, s in docs_with_scores if s >= self.r_score
                ]

            result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
            final_results = []
            for doc, doc_score in result[: self.top_n]:
                metadata = doc.metadata
                metadata["score"] = doc_score
                doc = Document(
                    page_content=doc.page_content,
                    metadata=metadata,
                )
                final_results.append(doc)
            return final_results
        else:
            log.warning(
                "No valid scores found, check your reranking function. Returning original documents."
            )
            return documents
