import json
import logging
import mimetypes
import os
import shutil
import asyncio

import re
import uuid
from datetime import datetime
from pathlib import Path
from functools import lru_cache
from typing import Iterator, List, Optional, Sequence, Union

from fastapi import (
    Depends,
    FastAPI,
    Query,
    File,
    Form,
    HTTPException,
    UploadFile,
    Request,
    status,
    APIRouter,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel


from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

from open_webui.models.files import FileModel, FileUpdateForm, Files
from open_webui.models.knowledge import KnowledgeForm, Knowledges
from open_webui.storage.provider import Storage


from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT

# Document loaders
from open_webui.retrieval.loaders.main import Loader
from open_webui.retrieval.loaders.youtube import YoutubeLoader

# Web search engines
from open_webui.retrieval.web.main import SearchResult
from open_webui.retrieval.web.utils import get_web_loader
from open_webui.retrieval.web.perplexity_search import search_perplexity_search
from open_webui.retrieval.web.brave import search_brave
from open_webui.retrieval.web.kagi import search_kagi
from open_webui.retrieval.web.mojeek import search_mojeek
from open_webui.retrieval.web.bocha import search_bocha
from open_webui.retrieval.web.duckduckgo import search_duckduckgo
from open_webui.retrieval.web.google_pse import search_google_pse
from open_webui.retrieval.web.jina_search import search_jina
from open_webui.retrieval.web.searchapi import search_searchapi
from open_webui.retrieval.web.serpapi import search_serpapi
from open_webui.retrieval.web.searxng import search_searxng
from open_webui.retrieval.web.yacy import search_yacy
from open_webui.retrieval.web.serper import search_serper
from open_webui.retrieval.web.serply import search_serply
from open_webui.retrieval.web.serpstack import search_serpstack
from open_webui.retrieval.web.tavily import search_tavily
from open_webui.retrieval.web.bing import search_bing
from open_webui.retrieval.web.azure import search_azure
from open_webui.retrieval.web.exa import search_exa
from open_webui.retrieval.web.perplexity import search_perplexity
from open_webui.retrieval.web.sougou import search_sougou
from open_webui.retrieval.web.firecrawl import search_firecrawl
from open_webui.retrieval.web.external import search_external

from open_webui.retrieval.utils import (
    get_content_from_url,
    get_embedding_function,
    get_reranking_function,
    clear_bm25_index_cache,
    invalidate_bm25_collections,
    mark_bm25_collections_dirty,
    query_collection,
    query_collection_with_rag_pipeline,
    query_doc,
    query_doc_with_rag_pipeline,
    merge_and_sort_query_results,
)
from open_webui.retrieval.vector.utils import filter_metadata
from open_webui.utils.misc import (
    calculate_sha256_string,
    sanitize_text_for_db,
)
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.file_upload_settings import (
    is_conversation_file_upload_embedding_enabled,
)

from open_webui.config import (
    ENV,
    VOYAGE_TOKENIZER_MODEL as DEFAULT_VOYAGE_TOKENIZER_MODEL,
    UPLOAD_DIR,
    KNOWLEDGE_UPLOAD_DIR,
    DEFAULT_LOCALE,
    RAG_EMBEDDING_CONTENT_PREFIX,
    RAG_EMBEDDING_QUERY_PREFIX,
)
from open_webui.env import (
    DEVICE_TYPE,
)

from open_webui.constants import ERROR_MESSAGES
from open_webui.utils.knowledge import (
    get_active_vector_collection_name,
    resolve_collection_rag_config,
)

log = logging.getLogger(__name__)


SUPPORTED_RERANKING_ENGINES = {"external", "voyage"}

##########################################
#
# Utility functions
#
##########################################


@lru_cache(maxsize=8)
def get_tiktoken_encoder(encoding_name: str):
    import tiktoken

    return tiktoken.get_encoding(encoding_name)


@lru_cache(maxsize=4)
def get_voyage_tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, use_fast=True)


def warm_voyage_tokenizer(model: str) -> None:
    try:
        tokenizer = get_voyage_tokenizer(model)
        tokenizer("warmup", add_special_tokens=False)
        log.info(f"Warmed voyage tokenizer: {model}")
    except Exception as e:
        log.warning(f"Failed to warm voyage tokenizer: {e}")


def _normalize_voyage_tokenizer_model(model: Optional[str]) -> str:
    normalized_model = (model or "").strip()
    if not normalized_model:
        return DEFAULT_VOYAGE_TOKENIZER_MODEL.value

    return normalized_model


def _should_warm_voyage_tokenizer(
    previous_text_splitter: str,
    current_text_splitter: str,
    previous_voyage_tokenizer_model_raw: Optional[str],
    current_voyage_tokenizer_model_raw: Optional[str],
) -> tuple[bool, bool, bool]:
    splitter_enabled = (
        previous_text_splitter != "token_voyage"
        and current_text_splitter == "token_voyage"
    )
    voyage_model_changed = _normalize_voyage_tokenizer_model(
        previous_voyage_tokenizer_model_raw
    ) != _normalize_voyage_tokenizer_model(current_voyage_tokenizer_model_raw)

    return splitter_enabled or voyage_model_changed, splitter_enabled, voyage_model_changed


def get_ef(
    engine: str,
    embedding_model: str,
):
    return None


def get_rf(
    engine: str = "",
    reranking_model: Optional[str] = None,
    external_reranker_url: str = "",
    external_reranker_api_key: str = "",
    external_reranker_timeout: str = "",
    voyage_reranker_url: str = "",
    voyage_reranker_api_key: str = "",
    voyage_reranker_timeout: str = "",
):
    rf = None

    if reranking_model:
        if engine not in SUPPORTED_RERANKING_ENGINES:
            raise ValueError(
                "RAG_RERANKING_ENGINE must be one of: external, voyage"
            )

        try:
            if engine == "external":
                # Convert timeout string to int or None (system default)
                timeout_value = (
                    int(external_reranker_timeout)
                    if external_reranker_timeout
                    else None
                )
                from open_webui.retrieval.models.external import ExternalReranker

                rf = ExternalReranker(
                    url=external_reranker_url,
                    api_key=external_reranker_api_key,
                    model=reranking_model,
                    timeout=timeout_value,
                )
            elif engine == "voyage":
                # Convert timeout string to int or None (system default)
                timeout_value = (
                    int(voyage_reranker_timeout) if voyage_reranker_timeout else None
                )
                from open_webui.retrieval.models.voyage import VoyageReranker

                rf = VoyageReranker(
                    url=voyage_reranker_url,
                    api_key=voyage_reranker_api_key,
                    model=reranking_model,
                    timeout=timeout_value,
                )
        except Exception as e:
            log.error(f"Reranking ({engine}): {e}")
            raise Exception(ERROR_MESSAGES.DEFAULT(e))

    return rf


def get_collection_effective_config(request: Request, collection_name: str | None) -> dict:
    if not collection_name:
        return resolve_collection_rag_config(None, request.app.state.config)["effective"]

    knowledge = Knowledges.get_knowledge_by_id(collection_name)
    if not knowledge:
        for kb in Knowledges.get_knowledge_bases():
            if (
                get_active_vector_collection_name(kb.id, kb.meta)
                == collection_name
            ):
                knowledge = kb
                break

    if not knowledge:
        return resolve_collection_rag_config(None, request.app.state.config)["effective"]

    return resolve_collection_rag_config(knowledge.meta, request.app.state.config)["effective"]


def get_physical_collection_name(collection_name: str | None) -> str | None:
    if not collection_name:
        return None

    knowledge = Knowledges.get_knowledge_by_id(collection_name)
    if not knowledge:
        return collection_name

    return get_active_vector_collection_name(knowledge.id, knowledge.meta)


def build_embedding_function_from_effective_config(request: Request, effective_config: dict):
    engine = effective_config["RAG_EMBEDDING_ENGINE"]
    return get_embedding_function(
        engine,
        effective_config["RAG_EMBEDDING_MODEL"],
        request.app.state.ef,
        (
            request.app.state.config.RAG_OPENAI_API_BASE_URL
            if engine == "openai"
            else request.app.state.config.RAG_AZURE_OPENAI_BASE_URL
        ),
        (
            request.app.state.config.RAG_OPENAI_API_KEY
            if engine == "openai"
            else request.app.state.config.RAG_AZURE_OPENAI_API_KEY
        ),
        effective_config["RAG_EMBEDDING_BATCH_SIZE"],
        azure_api_version=(
            request.app.state.config.RAG_AZURE_OPENAI_API_VERSION
            if engine == "azure_openai"
            else None
        ),
        enable_async=request.app.state.config.ENABLE_ASYNC_EMBEDDING,
    )


def build_reranking_function_from_effective_config(request: Request, effective_config: dict):
    model = effective_config.get("RAG_RERANKING_MODEL", "")
    if not model:
        return None

    engine = effective_config.get("RAG_RERANKING_ENGINE", "external")
    rf = get_rf(
        engine,
        model,
        request.app.state.config.RAG_EXTERNAL_RERANKER_URL,
        request.app.state.config.RAG_EXTERNAL_RERANKER_API_KEY,
        request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT,
        request.app.state.config.RAG_VOYAGE_RERANKER_URL,
        request.app.state.config.RAG_VOYAGE_RERANKER_API_KEY,
        request.app.state.config.RAG_VOYAGE_RERANKER_TIMEOUT,
    )
    return get_reranking_function(engine, model, rf)


##########################################
#
# API routes
#
##########################################


router = APIRouter()


class CollectionNameForm(BaseModel):
    collection_name: Optional[str] = None


class ProcessUrlForm(CollectionNameForm):
    url: str


class SearchForm(BaseModel):
    queries: List[str]


@router.get("/")
async def get_status(request: Request):
    return {
        "status": True,
        "CHUNK_SIZE": request.app.state.config.CHUNK_SIZE,
        "CHUNK_OVERLAP": request.app.state.config.CHUNK_OVERLAP,
        "RAG_TEMPLATE": request.app.state.config.RAG_TEMPLATE,
        "RAG_EMBEDDING_ENGINE": request.app.state.config.RAG_EMBEDDING_ENGINE,
        "RAG_EMBEDDING_MODEL": request.app.state.config.RAG_EMBEDDING_MODEL,
        "RAG_RERANKING_MODEL": request.app.state.config.RAG_RERANKING_MODEL,
        "RAG_EMBEDDING_BATCH_SIZE": request.app.state.config.RAG_EMBEDDING_BATCH_SIZE,
        "ENABLE_ASYNC_EMBEDDING": request.app.state.config.ENABLE_ASYNC_EMBEDDING,
    }


@router.get("/embedding")
async def get_embedding_config(request: Request, user=Depends(get_admin_user)):
    return {
        "status": True,
        "RAG_EMBEDDING_ENGINE": request.app.state.config.RAG_EMBEDDING_ENGINE,
        "RAG_EMBEDDING_MODEL": request.app.state.config.RAG_EMBEDDING_MODEL,
        "RAG_EMBEDDING_BATCH_SIZE": request.app.state.config.RAG_EMBEDDING_BATCH_SIZE,
        "ENABLE_ASYNC_EMBEDDING": request.app.state.config.ENABLE_ASYNC_EMBEDDING,
        "openai_config": {
            "url": request.app.state.config.RAG_OPENAI_API_BASE_URL,
            "key": request.app.state.config.RAG_OPENAI_API_KEY,
        },
        "azure_openai_config": {
            "url": request.app.state.config.RAG_AZURE_OPENAI_BASE_URL,
            "key": request.app.state.config.RAG_AZURE_OPENAI_API_KEY,
            "version": request.app.state.config.RAG_AZURE_OPENAI_API_VERSION,
        },
    }


class OpenAIConfigForm(BaseModel):
    url: str
    key: str


class AzureOpenAIConfigForm(BaseModel):
    url: str
    key: str
    version: str


class EmbeddingModelUpdateForm(BaseModel):
    openai_config: Optional[OpenAIConfigForm] = None
    azure_openai_config: Optional[AzureOpenAIConfigForm] = None
    RAG_EMBEDDING_ENGINE: str
    RAG_EMBEDDING_MODEL: str
    RAG_EMBEDDING_BATCH_SIZE: Optional[int] = 1
    ENABLE_ASYNC_EMBEDDING: Optional[bool] = True


def unload_embedding_model(request: Request):
    if request.app.state.config.RAG_EMBEDDING_ENGINE == "":
        # unloads current internal embedding model and clears VRAM cache
        request.app.state.ef = None
        request.app.state.EMBEDDING_FUNCTION = None
        import gc

        gc.collect()
        if DEVICE_TYPE == "cuda":
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@router.post("/embedding/update")
async def update_embedding_config(
    request: Request, form_data: EmbeddingModelUpdateForm, user=Depends(get_admin_user)
):
    log.info(
        f"Updating embedding model: {request.app.state.config.RAG_EMBEDDING_MODEL} to {form_data.RAG_EMBEDDING_MODEL}"
    )
    unload_embedding_model(request)
    try:
        request.app.state.config.RAG_EMBEDDING_ENGINE = form_data.RAG_EMBEDDING_ENGINE
        request.app.state.config.RAG_EMBEDDING_MODEL = form_data.RAG_EMBEDDING_MODEL
        request.app.state.config.RAG_EMBEDDING_BATCH_SIZE = (
            form_data.RAG_EMBEDDING_BATCH_SIZE
        )
        request.app.state.config.ENABLE_ASYNC_EMBEDDING = (
            form_data.ENABLE_ASYNC_EMBEDDING
        )

        if request.app.state.config.RAG_EMBEDDING_ENGINE in [
            "openai",
            "azure_openai",
        ]:
            if form_data.openai_config is not None:
                request.app.state.config.RAG_OPENAI_API_BASE_URL = (
                    form_data.openai_config.url
                )
                request.app.state.config.RAG_OPENAI_API_KEY = (
                    form_data.openai_config.key
                )

            if form_data.azure_openai_config is not None:
                request.app.state.config.RAG_AZURE_OPENAI_BASE_URL = (
                    form_data.azure_openai_config.url
                )
                request.app.state.config.RAG_AZURE_OPENAI_API_KEY = (
                    form_data.azure_openai_config.key
                )
                request.app.state.config.RAG_AZURE_OPENAI_API_VERSION = (
                    form_data.azure_openai_config.version
                )

        request.app.state.ef = get_ef(
            request.app.state.config.RAG_EMBEDDING_ENGINE,
            request.app.state.config.RAG_EMBEDDING_MODEL,
        )

        request.app.state.EMBEDDING_FUNCTION = get_embedding_function(
            request.app.state.config.RAG_EMBEDDING_ENGINE,
            request.app.state.config.RAG_EMBEDDING_MODEL,
            request.app.state.ef,
            (
                request.app.state.config.RAG_OPENAI_API_BASE_URL
                if request.app.state.config.RAG_EMBEDDING_ENGINE == "openai"
                else request.app.state.config.RAG_AZURE_OPENAI_BASE_URL
            ),
            (
                request.app.state.config.RAG_OPENAI_API_KEY
                if request.app.state.config.RAG_EMBEDDING_ENGINE == "openai"
                else request.app.state.config.RAG_AZURE_OPENAI_API_KEY
            ),
            request.app.state.config.RAG_EMBEDDING_BATCH_SIZE,
            azure_api_version=(
                request.app.state.config.RAG_AZURE_OPENAI_API_VERSION
                if request.app.state.config.RAG_EMBEDDING_ENGINE == "azure_openai"
                else None
            ),
            enable_async=request.app.state.config.ENABLE_ASYNC_EMBEDDING,
        )

        return {
            "status": True,
            "RAG_EMBEDDING_ENGINE": request.app.state.config.RAG_EMBEDDING_ENGINE,
            "RAG_EMBEDDING_MODEL": request.app.state.config.RAG_EMBEDDING_MODEL,
            "RAG_EMBEDDING_BATCH_SIZE": request.app.state.config.RAG_EMBEDDING_BATCH_SIZE,
            "ENABLE_ASYNC_EMBEDDING": request.app.state.config.ENABLE_ASYNC_EMBEDDING,
            "openai_config": {
                "url": request.app.state.config.RAG_OPENAI_API_BASE_URL,
                "key": request.app.state.config.RAG_OPENAI_API_KEY,
            },
            "azure_openai_config": {
                "url": request.app.state.config.RAG_AZURE_OPENAI_BASE_URL,
                "key": request.app.state.config.RAG_AZURE_OPENAI_API_KEY,
                "version": request.app.state.config.RAG_AZURE_OPENAI_API_VERSION,
            },
        }
    except Exception as e:
        log.exception(f"Problem updating embedding model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@router.get("/config")
async def get_rag_config(request: Request, user=Depends(get_admin_user)):
    return {
        "status": True,
        # RAG settings
        "RAG_TEMPLATE": request.app.state.config.RAG_TEMPLATE,
        "TOP_K": request.app.state.config.TOP_K,
        "BYPASS_EMBEDDING_AND_RETRIEVAL": request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL,
        "RAG_FULL_CONTEXT": request.app.state.config.RAG_FULL_CONTEXT,
        # Retrieval settings
        "ENABLE_RAG_BM25_SEARCH": request.app.state.config.ENABLE_RAG_BM25_SEARCH,
        "ENABLE_RAG_BM25_ENRICHED_TEXTS": request.app.state.config.ENABLE_RAG_BM25_ENRICHED_TEXTS,
        "ENABLE_RAG_RERANKING": request.app.state.config.ENABLE_RAG_RERANKING,
        "TOP_K_RERANKER": request.app.state.config.TOP_K_RERANKER,
        "RELEVANCE_THRESHOLD": request.app.state.config.RELEVANCE_THRESHOLD,
        "BM25_WEIGHT": request.app.state.config.BM25_WEIGHT,
        # Content extraction settings
        "CONTENT_EXTRACTION_ENGINE": request.app.state.config.CONTENT_EXTRACTION_ENGINE,
        "PDF_EXTRACT_IMAGES": request.app.state.config.PDF_EXTRACT_IMAGES,
        "CONVERSATION_FILE_UPLOAD_EMBEDDING": request.app.state.config.CONVERSATION_FILE_UPLOAD_EMBEDDING,
        "DATALAB_MARKER_API_KEY": request.app.state.config.DATALAB_MARKER_API_KEY,
        "DATALAB_MARKER_API_BASE_URL": request.app.state.config.DATALAB_MARKER_API_BASE_URL,
        "DATALAB_MARKER_ADDITIONAL_CONFIG": request.app.state.config.DATALAB_MARKER_ADDITIONAL_CONFIG,
        "DATALAB_MARKER_SKIP_CACHE": request.app.state.config.DATALAB_MARKER_SKIP_CACHE,
        "DATALAB_MARKER_FORCE_OCR": request.app.state.config.DATALAB_MARKER_FORCE_OCR,
        "DATALAB_MARKER_PAGINATE": request.app.state.config.DATALAB_MARKER_PAGINATE,
        "DATALAB_MARKER_STRIP_EXISTING_OCR": request.app.state.config.DATALAB_MARKER_STRIP_EXISTING_OCR,
        "DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION": request.app.state.config.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION,
        "DATALAB_MARKER_FORMAT_LINES": request.app.state.config.DATALAB_MARKER_FORMAT_LINES,
        "DATALAB_MARKER_USE_LLM": request.app.state.config.DATALAB_MARKER_USE_LLM,
        "DATALAB_MARKER_OUTPUT_FORMAT": request.app.state.config.DATALAB_MARKER_OUTPUT_FORMAT,
        "EXTERNAL_DOCUMENT_LOADER_URL": request.app.state.config.EXTERNAL_DOCUMENT_LOADER_URL,
        "EXTERNAL_DOCUMENT_LOADER_API_KEY": request.app.state.config.EXTERNAL_DOCUMENT_LOADER_API_KEY,
        "TIKA_SERVER_URL": request.app.state.config.TIKA_SERVER_URL,
        "DOCLING_SERVER_URL": request.app.state.config.DOCLING_SERVER_URL,
        "DOCLING_API_KEY": request.app.state.config.DOCLING_API_KEY,
        "DOCLING_PARAMS": request.app.state.config.DOCLING_PARAMS,
        "DOCUMENT_INTELLIGENCE_ENDPOINT": request.app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT,
        "DOCUMENT_INTELLIGENCE_KEY": request.app.state.config.DOCUMENT_INTELLIGENCE_KEY,
        "DOCUMENT_INTELLIGENCE_MODEL": request.app.state.config.DOCUMENT_INTELLIGENCE_MODEL,
        "MISTRAL_OCR_API_BASE_URL": request.app.state.config.MISTRAL_OCR_API_BASE_URL,
        "MISTRAL_OCR_API_KEY": request.app.state.config.MISTRAL_OCR_API_KEY,
        # MinerU settings
        "MINERU_API_MODE": request.app.state.config.MINERU_API_MODE,
        "MINERU_API_URL": request.app.state.config.MINERU_API_URL,
        "MINERU_API_KEY": request.app.state.config.MINERU_API_KEY,
        "MINERU_API_TIMEOUT": request.app.state.config.MINERU_API_TIMEOUT,
        "MINERU_PARAMS": request.app.state.config.MINERU_PARAMS,
        # Reranking settings
        "RAG_RERANKING_MODEL": request.app.state.config.RAG_RERANKING_MODEL,
        "RAG_RERANKING_ENGINE": request.app.state.config.RAG_RERANKING_ENGINE,
        "RAG_EXTERNAL_RERANKER_URL": request.app.state.config.RAG_EXTERNAL_RERANKER_URL,
        "RAG_EXTERNAL_RERANKER_API_KEY": request.app.state.config.RAG_EXTERNAL_RERANKER_API_KEY,
        "RAG_EXTERNAL_RERANKER_TIMEOUT": request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT,
        "RAG_VOYAGE_RERANKER_URL": request.app.state.config.RAG_VOYAGE_RERANKER_URL,
        "RAG_VOYAGE_RERANKER_API_KEY": request.app.state.config.RAG_VOYAGE_RERANKER_API_KEY,
        "RAG_VOYAGE_RERANKER_TIMEOUT": request.app.state.config.RAG_VOYAGE_RERANKER_TIMEOUT,
        # Chunking settings
        "TEXT_SPLITTER": request.app.state.config.TEXT_SPLITTER,
        "VOYAGE_TOKENIZER_MODEL": request.app.state.config.VOYAGE_TOKENIZER_MODEL,
        "CHUNK_SIZE": request.app.state.config.CHUNK_SIZE,
        "CHUNK_OVERLAP": request.app.state.config.CHUNK_OVERLAP,
        # File upload settings
        "FILE_MAX_SIZE": request.app.state.config.FILE_MAX_SIZE,
        "FILE_MAX_COUNT": request.app.state.config.FILE_MAX_COUNT,
        "FILE_IMAGE_COMPRESSION_WIDTH": request.app.state.config.FILE_IMAGE_COMPRESSION_WIDTH,
        "FILE_IMAGE_COMPRESSION_HEIGHT": request.app.state.config.FILE_IMAGE_COMPRESSION_HEIGHT,
        "ALLOWED_FILE_EXTENSIONS": request.app.state.config.ALLOWED_FILE_EXTENSIONS,
        # Integration settings
        "ENABLE_GOOGLE_DRIVE_INTEGRATION": request.app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION,
        "ENABLE_ONEDRIVE_INTEGRATION": request.app.state.config.ENABLE_ONEDRIVE_INTEGRATION,
        # Web search settings
        "web": {
            "ENABLE_WEB_SEARCH": request.app.state.config.ENABLE_WEB_SEARCH,
            "WEB_SEARCH_ENGINE": request.app.state.config.WEB_SEARCH_ENGINE,
            "WEB_SEARCH_TRUST_ENV": request.app.state.config.WEB_SEARCH_TRUST_ENV,
            "WEB_SEARCH_RESULT_COUNT": request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            "WEB_SEARCH_CONCURRENT_REQUESTS": request.app.state.config.WEB_SEARCH_CONCURRENT_REQUESTS,
            "WEB_LOADER_CONCURRENT_REQUESTS": request.app.state.config.WEB_LOADER_CONCURRENT_REQUESTS,
            "WEB_SEARCH_DOMAIN_FILTER_LIST": request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            "BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL": request.app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL,
            "BYPASS_WEB_SEARCH_WEB_LOADER": request.app.state.config.BYPASS_WEB_SEARCH_WEB_LOADER,
            "SEARXNG_QUERY_URL": request.app.state.config.SEARXNG_QUERY_URL,
            "SEARXNG_LANGUAGE": request.app.state.config.SEARXNG_LANGUAGE,
            "YACY_QUERY_URL": request.app.state.config.YACY_QUERY_URL,
            "YACY_USERNAME": request.app.state.config.YACY_USERNAME,
            "YACY_PASSWORD": request.app.state.config.YACY_PASSWORD,
            "GOOGLE_PSE_API_KEY": request.app.state.config.GOOGLE_PSE_API_KEY,
            "GOOGLE_PSE_ENGINE_ID": request.app.state.config.GOOGLE_PSE_ENGINE_ID,
            "BRAVE_SEARCH_API_KEY": request.app.state.config.BRAVE_SEARCH_API_KEY,
            "KAGI_SEARCH_API_KEY": request.app.state.config.KAGI_SEARCH_API_KEY,
            "MOJEEK_SEARCH_API_KEY": request.app.state.config.MOJEEK_SEARCH_API_KEY,
            "BOCHA_SEARCH_API_KEY": request.app.state.config.BOCHA_SEARCH_API_KEY,
            "SERPSTACK_API_KEY": request.app.state.config.SERPSTACK_API_KEY,
            "SERPSTACK_HTTPS": request.app.state.config.SERPSTACK_HTTPS,
            "SERPER_API_KEY": request.app.state.config.SERPER_API_KEY,
            "SERPLY_API_KEY": request.app.state.config.SERPLY_API_KEY,
            "TAVILY_API_KEY": request.app.state.config.TAVILY_API_KEY,
            "SEARCHAPI_API_KEY": request.app.state.config.SEARCHAPI_API_KEY,
            "SEARCHAPI_ENGINE": request.app.state.config.SEARCHAPI_ENGINE,
            "SERPAPI_API_KEY": request.app.state.config.SERPAPI_API_KEY,
            "SERPAPI_ENGINE": request.app.state.config.SERPAPI_ENGINE,
            "JINA_API_KEY": request.app.state.config.JINA_API_KEY,
            "BING_SEARCH_V7_ENDPOINT": request.app.state.config.BING_SEARCH_V7_ENDPOINT,
            "BING_SEARCH_V7_SUBSCRIPTION_KEY": request.app.state.config.BING_SEARCH_V7_SUBSCRIPTION_KEY,
            "EXA_API_KEY": request.app.state.config.EXA_API_KEY,
            "PERPLEXITY_API_KEY": request.app.state.config.PERPLEXITY_API_KEY,
            "PERPLEXITY_MODEL": request.app.state.config.PERPLEXITY_MODEL,
            "PERPLEXITY_SEARCH_CONTEXT_USAGE": request.app.state.config.PERPLEXITY_SEARCH_CONTEXT_USAGE,
            "PERPLEXITY_SEARCH_API_URL": request.app.state.config.PERPLEXITY_SEARCH_API_URL,
            "SOUGOU_API_SID": request.app.state.config.SOUGOU_API_SID,
            "SOUGOU_API_SK": request.app.state.config.SOUGOU_API_SK,
            "WEB_LOADER_ENGINE": request.app.state.config.WEB_LOADER_ENGINE,
            "WEB_LOADER_TIMEOUT": request.app.state.config.WEB_LOADER_TIMEOUT,
            "ENABLE_WEB_LOADER_SSL_VERIFICATION": request.app.state.config.ENABLE_WEB_LOADER_SSL_VERIFICATION,
            "PLAYWRIGHT_WS_URL": request.app.state.config.PLAYWRIGHT_WS_URL,
            "PLAYWRIGHT_TIMEOUT": request.app.state.config.PLAYWRIGHT_TIMEOUT,
            "FIRECRAWL_API_KEY": request.app.state.config.FIRECRAWL_API_KEY,
            "FIRECRAWL_API_BASE_URL": request.app.state.config.FIRECRAWL_API_BASE_URL,
            "TAVILY_EXTRACT_DEPTH": request.app.state.config.TAVILY_EXTRACT_DEPTH,
            "EXTERNAL_WEB_SEARCH_URL": request.app.state.config.EXTERNAL_WEB_SEARCH_URL,
            "EXTERNAL_WEB_SEARCH_API_KEY": request.app.state.config.EXTERNAL_WEB_SEARCH_API_KEY,
            "EXTERNAL_WEB_LOADER_URL": request.app.state.config.EXTERNAL_WEB_LOADER_URL,
            "EXTERNAL_WEB_LOADER_API_KEY": request.app.state.config.EXTERNAL_WEB_LOADER_API_KEY,
            "YOUTUBE_LOADER_LANGUAGE": request.app.state.config.YOUTUBE_LOADER_LANGUAGE,
            "YOUTUBE_LOADER_PROXY_URL": request.app.state.config.YOUTUBE_LOADER_PROXY_URL,
            "YOUTUBE_LOADER_TRANSLATION": request.app.state.YOUTUBE_LOADER_TRANSLATION,
        },
    }


class WebConfig(BaseModel):
    ENABLE_WEB_SEARCH: Optional[bool] = None
    WEB_SEARCH_ENGINE: Optional[str] = None
    WEB_SEARCH_TRUST_ENV: Optional[bool] = None
    WEB_SEARCH_RESULT_COUNT: Optional[int] = None
    WEB_SEARCH_CONCURRENT_REQUESTS: Optional[int] = None
    WEB_LOADER_CONCURRENT_REQUESTS: Optional[int] = None
    WEB_SEARCH_DOMAIN_FILTER_LIST: Optional[List[str]] = []
    BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL: Optional[bool] = None
    BYPASS_WEB_SEARCH_WEB_LOADER: Optional[bool] = None
    SEARXNG_QUERY_URL: Optional[str] = None
    SEARXNG_LANGUAGE: Optional[str] = None
    YACY_QUERY_URL: Optional[str] = None
    YACY_USERNAME: Optional[str] = None
    YACY_PASSWORD: Optional[str] = None
    GOOGLE_PSE_API_KEY: Optional[str] = None
    GOOGLE_PSE_ENGINE_ID: Optional[str] = None
    BRAVE_SEARCH_API_KEY: Optional[str] = None
    KAGI_SEARCH_API_KEY: Optional[str] = None
    MOJEEK_SEARCH_API_KEY: Optional[str] = None
    BOCHA_SEARCH_API_KEY: Optional[str] = None
    SERPSTACK_API_KEY: Optional[str] = None
    SERPSTACK_HTTPS: Optional[bool] = None
    SERPER_API_KEY: Optional[str] = None
    SERPLY_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    SEARCHAPI_API_KEY: Optional[str] = None
    SEARCHAPI_ENGINE: Optional[str] = None
    SERPAPI_API_KEY: Optional[str] = None
    SERPAPI_ENGINE: Optional[str] = None
    JINA_API_KEY: Optional[str] = None
    BING_SEARCH_V7_ENDPOINT: Optional[str] = None
    BING_SEARCH_V7_SUBSCRIPTION_KEY: Optional[str] = None
    EXA_API_KEY: Optional[str] = None
    PERPLEXITY_API_KEY: Optional[str] = None
    PERPLEXITY_MODEL: Optional[str] = None
    PERPLEXITY_SEARCH_CONTEXT_USAGE: Optional[str] = None
    PERPLEXITY_SEARCH_API_URL: Optional[str] = None
    SOUGOU_API_SID: Optional[str] = None
    SOUGOU_API_SK: Optional[str] = None
    WEB_LOADER_ENGINE: Optional[str] = None
    WEB_LOADER_TIMEOUT: Optional[str] = None
    ENABLE_WEB_LOADER_SSL_VERIFICATION: Optional[bool] = None
    PLAYWRIGHT_WS_URL: Optional[str] = None
    PLAYWRIGHT_TIMEOUT: Optional[int] = None
    FIRECRAWL_API_KEY: Optional[str] = None
    FIRECRAWL_API_BASE_URL: Optional[str] = None
    TAVILY_EXTRACT_DEPTH: Optional[str] = None
    EXTERNAL_WEB_SEARCH_URL: Optional[str] = None
    EXTERNAL_WEB_SEARCH_API_KEY: Optional[str] = None
    EXTERNAL_WEB_LOADER_URL: Optional[str] = None
    EXTERNAL_WEB_LOADER_API_KEY: Optional[str] = None
    YOUTUBE_LOADER_LANGUAGE: Optional[List[str]] = None
    YOUTUBE_LOADER_PROXY_URL: Optional[str] = None
    YOUTUBE_LOADER_TRANSLATION: Optional[str] = None


class ConfigForm(BaseModel):
    # RAG settings
    RAG_TEMPLATE: Optional[str] = None
    TOP_K: Optional[int] = None
    BYPASS_EMBEDDING_AND_RETRIEVAL: Optional[bool] = None
    RAG_FULL_CONTEXT: Optional[bool] = None

    # Retrieval settings
    ENABLE_RAG_BM25_SEARCH: Optional[bool] = None
    ENABLE_RAG_BM25_ENRICHED_TEXTS: Optional[bool] = None
    ENABLE_RAG_RERANKING: Optional[bool] = None
    TOP_K_RERANKER: Optional[int] = None
    RELEVANCE_THRESHOLD: Optional[float] = None
    BM25_WEIGHT: Optional[float] = None

    # Content extraction settings
    CONTENT_EXTRACTION_ENGINE: Optional[str] = None
    PDF_EXTRACT_IMAGES: Optional[bool] = None
    CONVERSATION_FILE_UPLOAD_EMBEDDING: Optional[bool] = None

    DATALAB_MARKER_API_KEY: Optional[str] = None
    DATALAB_MARKER_API_BASE_URL: Optional[str] = None
    DATALAB_MARKER_ADDITIONAL_CONFIG: Optional[str] = None
    DATALAB_MARKER_SKIP_CACHE: Optional[bool] = None
    DATALAB_MARKER_FORCE_OCR: Optional[bool] = None
    DATALAB_MARKER_PAGINATE: Optional[bool] = None
    DATALAB_MARKER_STRIP_EXISTING_OCR: Optional[bool] = None
    DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION: Optional[bool] = None
    DATALAB_MARKER_FORMAT_LINES: Optional[bool] = None
    DATALAB_MARKER_USE_LLM: Optional[bool] = None
    DATALAB_MARKER_OUTPUT_FORMAT: Optional[str] = None

    EXTERNAL_DOCUMENT_LOADER_URL: Optional[str] = None
    EXTERNAL_DOCUMENT_LOADER_API_KEY: Optional[str] = None

    TIKA_SERVER_URL: Optional[str] = None
    DOCLING_SERVER_URL: Optional[str] = None
    DOCLING_API_KEY: Optional[str] = None
    DOCLING_PARAMS: Optional[dict] = None
    DOCUMENT_INTELLIGENCE_ENDPOINT: Optional[str] = None
    DOCUMENT_INTELLIGENCE_KEY: Optional[str] = None
    DOCUMENT_INTELLIGENCE_MODEL: Optional[str] = None
    MISTRAL_OCR_API_BASE_URL: Optional[str] = None
    MISTRAL_OCR_API_KEY: Optional[str] = None

    # MinerU settings
    MINERU_API_MODE: Optional[str] = None
    MINERU_API_URL: Optional[str] = None
    MINERU_API_KEY: Optional[str] = None
    MINERU_API_TIMEOUT: Optional[str] = None
    MINERU_PARAMS: Optional[dict] = None

    # Reranking settings
    RAG_RERANKING_MODEL: Optional[str] = None
    RAG_RERANKING_ENGINE: Optional[str] = None
    RAG_EXTERNAL_RERANKER_URL: Optional[str] = None
    RAG_EXTERNAL_RERANKER_API_KEY: Optional[str] = None
    RAG_EXTERNAL_RERANKER_TIMEOUT: Optional[str] = None
    RAG_VOYAGE_RERANKER_URL: Optional[str] = None
    RAG_VOYAGE_RERANKER_API_KEY: Optional[str] = None
    RAG_VOYAGE_RERANKER_TIMEOUT: Optional[str] = None

    # Chunking settings
    TEXT_SPLITTER: Optional[str] = None
    VOYAGE_TOKENIZER_MODEL: Optional[str] = None
    CHUNK_SIZE: Optional[int] = None
    CHUNK_OVERLAP: Optional[int] = None

    # File upload settings
    FILE_MAX_SIZE: Optional[int] = None
    FILE_MAX_COUNT: Optional[int] = None
    FILE_IMAGE_COMPRESSION_WIDTH: Optional[int] = None
    FILE_IMAGE_COMPRESSION_HEIGHT: Optional[int] = None
    ALLOWED_FILE_EXTENSIONS: Optional[List[str]] = None

    # Integration settings
    ENABLE_GOOGLE_DRIVE_INTEGRATION: Optional[bool] = None
    ENABLE_ONEDRIVE_INTEGRATION: Optional[bool] = None

    # Web search settings
    web: Optional[WebConfig] = None


@router.post("/config/update")
async def update_rag_config(
    request: Request, form_data: ConfigForm, user=Depends(get_admin_user)
):
    # RAG settings
    request.app.state.config.RAG_TEMPLATE = (
        form_data.RAG_TEMPLATE
        if form_data.RAG_TEMPLATE is not None
        else request.app.state.config.RAG_TEMPLATE
    )
    request.app.state.config.TOP_K = (
        form_data.TOP_K
        if form_data.TOP_K is not None
        else request.app.state.config.TOP_K
    )
    request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL = (
        form_data.BYPASS_EMBEDDING_AND_RETRIEVAL
        if form_data.BYPASS_EMBEDDING_AND_RETRIEVAL is not None
        else request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL
    )
    request.app.state.config.RAG_FULL_CONTEXT = (
        form_data.RAG_FULL_CONTEXT
        if form_data.RAG_FULL_CONTEXT is not None
        else request.app.state.config.RAG_FULL_CONTEXT
    )

    # Retrieval settings
    request.app.state.config.ENABLE_RAG_BM25_SEARCH = (
        form_data.ENABLE_RAG_BM25_SEARCH
        if form_data.ENABLE_RAG_BM25_SEARCH is not None
        else request.app.state.config.ENABLE_RAG_BM25_SEARCH
    )
    request.app.state.config.ENABLE_RAG_BM25_ENRICHED_TEXTS = (
        form_data.ENABLE_RAG_BM25_ENRICHED_TEXTS
        if form_data.ENABLE_RAG_BM25_ENRICHED_TEXTS is not None
        else request.app.state.config.ENABLE_RAG_BM25_ENRICHED_TEXTS
    )
    request.app.state.config.ENABLE_RAG_RERANKING = (
        form_data.ENABLE_RAG_RERANKING
        if form_data.ENABLE_RAG_RERANKING is not None
        else request.app.state.config.ENABLE_RAG_RERANKING
    )

    request.app.state.config.TOP_K_RERANKER = (
        form_data.TOP_K_RERANKER
        if form_data.TOP_K_RERANKER is not None
        else request.app.state.config.TOP_K_RERANKER
    )
    request.app.state.config.RELEVANCE_THRESHOLD = (
        form_data.RELEVANCE_THRESHOLD
        if form_data.RELEVANCE_THRESHOLD is not None
        else request.app.state.config.RELEVANCE_THRESHOLD
    )
    request.app.state.config.BM25_WEIGHT = (
        form_data.BM25_WEIGHT
        if form_data.BM25_WEIGHT is not None
        else request.app.state.config.BM25_WEIGHT
    )

    # Content extraction settings
    request.app.state.config.CONTENT_EXTRACTION_ENGINE = (
        form_data.CONTENT_EXTRACTION_ENGINE
        if form_data.CONTENT_EXTRACTION_ENGINE is not None
        else request.app.state.config.CONTENT_EXTRACTION_ENGINE
    )
    request.app.state.config.PDF_EXTRACT_IMAGES = (
        form_data.PDF_EXTRACT_IMAGES
        if form_data.PDF_EXTRACT_IMAGES is not None
        else request.app.state.config.PDF_EXTRACT_IMAGES
    )
    request.app.state.config.CONVERSATION_FILE_UPLOAD_EMBEDDING = (
        form_data.CONVERSATION_FILE_UPLOAD_EMBEDDING
        if form_data.CONVERSATION_FILE_UPLOAD_EMBEDDING is not None
        else request.app.state.config.CONVERSATION_FILE_UPLOAD_EMBEDDING
    )
    request.app.state.config.DATALAB_MARKER_API_KEY = (
        form_data.DATALAB_MARKER_API_KEY
        if form_data.DATALAB_MARKER_API_KEY is not None
        else request.app.state.config.DATALAB_MARKER_API_KEY
    )
    request.app.state.config.DATALAB_MARKER_API_BASE_URL = (
        form_data.DATALAB_MARKER_API_BASE_URL
        if form_data.DATALAB_MARKER_API_BASE_URL is not None
        else request.app.state.config.DATALAB_MARKER_API_BASE_URL
    )
    request.app.state.config.DATALAB_MARKER_ADDITIONAL_CONFIG = (
        form_data.DATALAB_MARKER_ADDITIONAL_CONFIG
        if form_data.DATALAB_MARKER_ADDITIONAL_CONFIG is not None
        else request.app.state.config.DATALAB_MARKER_ADDITIONAL_CONFIG
    )
    request.app.state.config.DATALAB_MARKER_SKIP_CACHE = (
        form_data.DATALAB_MARKER_SKIP_CACHE
        if form_data.DATALAB_MARKER_SKIP_CACHE is not None
        else request.app.state.config.DATALAB_MARKER_SKIP_CACHE
    )
    request.app.state.config.DATALAB_MARKER_FORCE_OCR = (
        form_data.DATALAB_MARKER_FORCE_OCR
        if form_data.DATALAB_MARKER_FORCE_OCR is not None
        else request.app.state.config.DATALAB_MARKER_FORCE_OCR
    )
    request.app.state.config.DATALAB_MARKER_PAGINATE = (
        form_data.DATALAB_MARKER_PAGINATE
        if form_data.DATALAB_MARKER_PAGINATE is not None
        else request.app.state.config.DATALAB_MARKER_PAGINATE
    )
    request.app.state.config.DATALAB_MARKER_STRIP_EXISTING_OCR = (
        form_data.DATALAB_MARKER_STRIP_EXISTING_OCR
        if form_data.DATALAB_MARKER_STRIP_EXISTING_OCR is not None
        else request.app.state.config.DATALAB_MARKER_STRIP_EXISTING_OCR
    )
    request.app.state.config.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION = (
        form_data.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION
        if form_data.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION is not None
        else request.app.state.config.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION
    )
    request.app.state.config.DATALAB_MARKER_FORMAT_LINES = (
        form_data.DATALAB_MARKER_FORMAT_LINES
        if form_data.DATALAB_MARKER_FORMAT_LINES is not None
        else request.app.state.config.DATALAB_MARKER_FORMAT_LINES
    )
    request.app.state.config.DATALAB_MARKER_OUTPUT_FORMAT = (
        form_data.DATALAB_MARKER_OUTPUT_FORMAT
        if form_data.DATALAB_MARKER_OUTPUT_FORMAT is not None
        else request.app.state.config.DATALAB_MARKER_OUTPUT_FORMAT
    )
    request.app.state.config.DATALAB_MARKER_USE_LLM = (
        form_data.DATALAB_MARKER_USE_LLM
        if form_data.DATALAB_MARKER_USE_LLM is not None
        else request.app.state.config.DATALAB_MARKER_USE_LLM
    )
    request.app.state.config.EXTERNAL_DOCUMENT_LOADER_URL = (
        form_data.EXTERNAL_DOCUMENT_LOADER_URL
        if form_data.EXTERNAL_DOCUMENT_LOADER_URL is not None
        else request.app.state.config.EXTERNAL_DOCUMENT_LOADER_URL
    )
    request.app.state.config.EXTERNAL_DOCUMENT_LOADER_API_KEY = (
        form_data.EXTERNAL_DOCUMENT_LOADER_API_KEY
        if form_data.EXTERNAL_DOCUMENT_LOADER_API_KEY is not None
        else request.app.state.config.EXTERNAL_DOCUMENT_LOADER_API_KEY
    )
    request.app.state.config.TIKA_SERVER_URL = (
        form_data.TIKA_SERVER_URL
        if form_data.TIKA_SERVER_URL is not None
        else request.app.state.config.TIKA_SERVER_URL
    )
    request.app.state.config.DOCLING_SERVER_URL = (
        form_data.DOCLING_SERVER_URL
        if form_data.DOCLING_SERVER_URL is not None
        else request.app.state.config.DOCLING_SERVER_URL
    )
    request.app.state.config.DOCLING_API_KEY = (
        form_data.DOCLING_API_KEY
        if form_data.DOCLING_API_KEY is not None
        else request.app.state.config.DOCLING_API_KEY
    )
    request.app.state.config.DOCLING_PARAMS = (
        form_data.DOCLING_PARAMS
        if form_data.DOCLING_PARAMS is not None
        else request.app.state.config.DOCLING_PARAMS
    )
    request.app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT = (
        form_data.DOCUMENT_INTELLIGENCE_ENDPOINT
        if form_data.DOCUMENT_INTELLIGENCE_ENDPOINT is not None
        else request.app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT
    )
    request.app.state.config.DOCUMENT_INTELLIGENCE_KEY = (
        form_data.DOCUMENT_INTELLIGENCE_KEY
        if form_data.DOCUMENT_INTELLIGENCE_KEY is not None
        else request.app.state.config.DOCUMENT_INTELLIGENCE_KEY
    )
    request.app.state.config.DOCUMENT_INTELLIGENCE_MODEL = (
        form_data.DOCUMENT_INTELLIGENCE_MODEL
        if form_data.DOCUMENT_INTELLIGENCE_MODEL is not None
        else request.app.state.config.DOCUMENT_INTELLIGENCE_MODEL
    )

    request.app.state.config.MISTRAL_OCR_API_BASE_URL = (
        form_data.MISTRAL_OCR_API_BASE_URL
        if form_data.MISTRAL_OCR_API_BASE_URL is not None
        else request.app.state.config.MISTRAL_OCR_API_BASE_URL
    )
    request.app.state.config.MISTRAL_OCR_API_KEY = (
        form_data.MISTRAL_OCR_API_KEY
        if form_data.MISTRAL_OCR_API_KEY is not None
        else request.app.state.config.MISTRAL_OCR_API_KEY
    )

    # MinerU settings
    request.app.state.config.MINERU_API_MODE = (
        form_data.MINERU_API_MODE
        if form_data.MINERU_API_MODE is not None
        else request.app.state.config.MINERU_API_MODE
    )
    request.app.state.config.MINERU_API_URL = (
        form_data.MINERU_API_URL
        if form_data.MINERU_API_URL is not None
        else request.app.state.config.MINERU_API_URL
    )
    request.app.state.config.MINERU_API_KEY = (
        form_data.MINERU_API_KEY
        if form_data.MINERU_API_KEY is not None
        else request.app.state.config.MINERU_API_KEY
    )
    request.app.state.config.MINERU_API_TIMEOUT = (
        form_data.MINERU_API_TIMEOUT
        if form_data.MINERU_API_TIMEOUT is not None
        else request.app.state.config.MINERU_API_TIMEOUT
    )
    request.app.state.config.MINERU_PARAMS = (
        form_data.MINERU_PARAMS
        if form_data.MINERU_PARAMS is not None
        else request.app.state.config.MINERU_PARAMS
    )

    # Reranking settings
    reranking_engine = (
        form_data.RAG_RERANKING_ENGINE
        if form_data.RAG_RERANKING_ENGINE is not None
        else request.app.state.config.RAG_RERANKING_ENGINE
    )
    if reranking_engine not in SUPPORTED_RERANKING_ENGINES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="RAG_RERANKING_ENGINE must be one of: external, voyage",
        )

    request.app.state.config.RAG_RERANKING_ENGINE = reranking_engine

    request.app.state.config.RAG_EXTERNAL_RERANKER_URL = (
        form_data.RAG_EXTERNAL_RERANKER_URL
        if form_data.RAG_EXTERNAL_RERANKER_URL is not None
        else request.app.state.config.RAG_EXTERNAL_RERANKER_URL
    )

    request.app.state.config.RAG_EXTERNAL_RERANKER_API_KEY = (
        form_data.RAG_EXTERNAL_RERANKER_API_KEY
        if form_data.RAG_EXTERNAL_RERANKER_API_KEY is not None
        else request.app.state.config.RAG_EXTERNAL_RERANKER_API_KEY
    )

    request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT = (
        form_data.RAG_EXTERNAL_RERANKER_TIMEOUT
        if form_data.RAG_EXTERNAL_RERANKER_TIMEOUT is not None
        else request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT
    )

    request.app.state.config.RAG_VOYAGE_RERANKER_URL = (
        form_data.RAG_VOYAGE_RERANKER_URL
        if form_data.RAG_VOYAGE_RERANKER_URL is not None
        else request.app.state.config.RAG_VOYAGE_RERANKER_URL
    )

    request.app.state.config.RAG_VOYAGE_RERANKER_API_KEY = (
        form_data.RAG_VOYAGE_RERANKER_API_KEY
        if form_data.RAG_VOYAGE_RERANKER_API_KEY is not None
        else request.app.state.config.RAG_VOYAGE_RERANKER_API_KEY
    )

    request.app.state.config.RAG_VOYAGE_RERANKER_TIMEOUT = (
        form_data.RAG_VOYAGE_RERANKER_TIMEOUT
        if form_data.RAG_VOYAGE_RERANKER_TIMEOUT is not None
        else request.app.state.config.RAG_VOYAGE_RERANKER_TIMEOUT
    )

    if not request.app.state.config.ENABLE_RAG_RERANKING:
        # Unload current reranker and clear VRAM cache.
        request.app.state.rf = None
        request.app.state.RERANKING_FUNCTION = None
        import gc

        gc.collect()
        if DEVICE_TYPE == "cuda":
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    log.info(
        f"Updating reranking model: {request.app.state.config.RAG_RERANKING_MODEL} to {form_data.RAG_RERANKING_MODEL}"
    )
    try:
        request.app.state.config.RAG_RERANKING_MODEL = (
            form_data.RAG_RERANKING_MODEL
            if form_data.RAG_RERANKING_MODEL is not None
            else request.app.state.config.RAG_RERANKING_MODEL
        )

        try:
            if (
                request.app.state.config.ENABLE_RAG_RERANKING
                and not request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL
            ):
                request.app.state.rf = get_rf(
                    request.app.state.config.RAG_RERANKING_ENGINE,
                    request.app.state.config.RAG_RERANKING_MODEL,
                    request.app.state.config.RAG_EXTERNAL_RERANKER_URL,
                    request.app.state.config.RAG_EXTERNAL_RERANKER_API_KEY,
                    request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT,
                    request.app.state.config.RAG_VOYAGE_RERANKER_URL,
                    request.app.state.config.RAG_VOYAGE_RERANKER_API_KEY,
                    request.app.state.config.RAG_VOYAGE_RERANKER_TIMEOUT,
                )

                request.app.state.RERANKING_FUNCTION = get_reranking_function(
                    request.app.state.config.RAG_RERANKING_ENGINE,
                    request.app.state.config.RAG_RERANKING_MODEL,
                    request.app.state.rf,
                )
        except Exception as e:
            log.error(f"Error loading reranking model: {e}")
            request.app.state.config.ENABLE_RAG_RERANKING = False
    except Exception as e:
        log.exception(f"Problem updating reranking model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )

    # Chunking settings
    previous_text_splitter = request.app.state.config.TEXT_SPLITTER
    previous_voyage_tokenizer_model_raw = request.app.state.config.VOYAGE_TOKENIZER_MODEL
    request.app.state.config.TEXT_SPLITTER = (
        form_data.TEXT_SPLITTER
        if form_data.TEXT_SPLITTER is not None
        else request.app.state.config.TEXT_SPLITTER
    )
    request.app.state.config.VOYAGE_TOKENIZER_MODEL = (
        form_data.VOYAGE_TOKENIZER_MODEL
        if form_data.VOYAGE_TOKENIZER_MODEL is not None
        else request.app.state.config.VOYAGE_TOKENIZER_MODEL
    )
    current_text_splitter = request.app.state.config.TEXT_SPLITTER
    current_voyage_tokenizer_model_raw = request.app.state.config.VOYAGE_TOKENIZER_MODEL

    if (
        current_text_splitter == "token_voyage"
        and not (current_voyage_tokenizer_model_raw or "").strip()
    ):
        request.app.state.config.VOYAGE_TOKENIZER_MODEL = (
            DEFAULT_VOYAGE_TOKENIZER_MODEL.value
        )
        current_voyage_tokenizer_model_raw = (
            request.app.state.config.VOYAGE_TOKENIZER_MODEL
        )
    request.app.state.config.CHUNK_SIZE = (
        form_data.CHUNK_SIZE
        if form_data.CHUNK_SIZE is not None
        else request.app.state.config.CHUNK_SIZE
    )
    request.app.state.config.CHUNK_OVERLAP = (
        form_data.CHUNK_OVERLAP
        if form_data.CHUNK_OVERLAP is not None
        else request.app.state.config.CHUNK_OVERLAP
    )
    (
        should_warm_voyage_tokenizer,
        splitter_enabled,
        voyage_model_changed,
    ) = _should_warm_voyage_tokenizer(
        previous_text_splitter=previous_text_splitter,
        current_text_splitter=current_text_splitter,
        previous_voyage_tokenizer_model_raw=previous_voyage_tokenizer_model_raw,
        current_voyage_tokenizer_model_raw=current_voyage_tokenizer_model_raw,
    )
    target_voyage_tokenizer_model = _normalize_voyage_tokenizer_model(
        current_voyage_tokenizer_model_raw
    )

    if should_warm_voyage_tokenizer:
        reasons = []
        if splitter_enabled:
            reasons.append("splitter_enabled")
        if voyage_model_changed:
            reasons.append("model_changed")

        log.info(
            "Scheduling voyage tokenizer warmup: "
            f"model={target_voyage_tokenizer_model}, reason={','.join(reasons)}"
        )
        asyncio.create_task(
            run_in_threadpool(
                warm_voyage_tokenizer,
                target_voyage_tokenizer_model,
            )
        )

    # File upload settings
    request.app.state.config.FILE_MAX_SIZE = form_data.FILE_MAX_SIZE
    request.app.state.config.FILE_MAX_COUNT = form_data.FILE_MAX_COUNT
    request.app.state.config.FILE_IMAGE_COMPRESSION_WIDTH = (
        form_data.FILE_IMAGE_COMPRESSION_WIDTH
    )
    request.app.state.config.FILE_IMAGE_COMPRESSION_HEIGHT = (
        form_data.FILE_IMAGE_COMPRESSION_HEIGHT
    )
    request.app.state.config.ALLOWED_FILE_EXTENSIONS = (
        form_data.ALLOWED_FILE_EXTENSIONS
        if form_data.ALLOWED_FILE_EXTENSIONS is not None
        else request.app.state.config.ALLOWED_FILE_EXTENSIONS
    )

    # Integration settings
    request.app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION = (
        form_data.ENABLE_GOOGLE_DRIVE_INTEGRATION
        if form_data.ENABLE_GOOGLE_DRIVE_INTEGRATION is not None
        else request.app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION
    )
    request.app.state.config.ENABLE_ONEDRIVE_INTEGRATION = (
        form_data.ENABLE_ONEDRIVE_INTEGRATION
        if form_data.ENABLE_ONEDRIVE_INTEGRATION is not None
        else request.app.state.config.ENABLE_ONEDRIVE_INTEGRATION
    )

    if form_data.web is not None:
        # Web search settings
        request.app.state.config.ENABLE_WEB_SEARCH = form_data.web.ENABLE_WEB_SEARCH
        request.app.state.config.WEB_SEARCH_ENGINE = form_data.web.WEB_SEARCH_ENGINE
        request.app.state.config.WEB_SEARCH_TRUST_ENV = (
            form_data.web.WEB_SEARCH_TRUST_ENV
        )
        request.app.state.config.WEB_SEARCH_RESULT_COUNT = (
            form_data.web.WEB_SEARCH_RESULT_COUNT
        )
        request.app.state.config.WEB_SEARCH_CONCURRENT_REQUESTS = (
            form_data.web.WEB_SEARCH_CONCURRENT_REQUESTS
        )
        request.app.state.config.WEB_LOADER_CONCURRENT_REQUESTS = (
            form_data.web.WEB_LOADER_CONCURRENT_REQUESTS
        )
        request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST = (
            form_data.web.WEB_SEARCH_DOMAIN_FILTER_LIST
        )
        request.app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL = (
            form_data.web.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL
        )
        request.app.state.config.BYPASS_WEB_SEARCH_WEB_LOADER = (
            form_data.web.BYPASS_WEB_SEARCH_WEB_LOADER
        )
        request.app.state.config.SEARXNG_QUERY_URL = form_data.web.SEARXNG_QUERY_URL
        request.app.state.config.SEARXNG_LANGUAGE = form_data.web.SEARXNG_LANGUAGE
        request.app.state.config.YACY_QUERY_URL = form_data.web.YACY_QUERY_URL
        request.app.state.config.YACY_USERNAME = form_data.web.YACY_USERNAME
        request.app.state.config.YACY_PASSWORD = form_data.web.YACY_PASSWORD
        request.app.state.config.GOOGLE_PSE_API_KEY = form_data.web.GOOGLE_PSE_API_KEY
        request.app.state.config.GOOGLE_PSE_ENGINE_ID = (
            form_data.web.GOOGLE_PSE_ENGINE_ID
        )
        request.app.state.config.BRAVE_SEARCH_API_KEY = (
            form_data.web.BRAVE_SEARCH_API_KEY
        )
        request.app.state.config.KAGI_SEARCH_API_KEY = form_data.web.KAGI_SEARCH_API_KEY
        request.app.state.config.MOJEEK_SEARCH_API_KEY = (
            form_data.web.MOJEEK_SEARCH_API_KEY
        )
        request.app.state.config.BOCHA_SEARCH_API_KEY = (
            form_data.web.BOCHA_SEARCH_API_KEY
        )
        request.app.state.config.SERPSTACK_API_KEY = form_data.web.SERPSTACK_API_KEY
        request.app.state.config.SERPSTACK_HTTPS = form_data.web.SERPSTACK_HTTPS
        request.app.state.config.SERPER_API_KEY = form_data.web.SERPER_API_KEY
        request.app.state.config.SERPLY_API_KEY = form_data.web.SERPLY_API_KEY
        request.app.state.config.TAVILY_API_KEY = form_data.web.TAVILY_API_KEY
        request.app.state.config.SEARCHAPI_API_KEY = form_data.web.SEARCHAPI_API_KEY
        request.app.state.config.SEARCHAPI_ENGINE = form_data.web.SEARCHAPI_ENGINE
        request.app.state.config.SERPAPI_API_KEY = form_data.web.SERPAPI_API_KEY
        request.app.state.config.SERPAPI_ENGINE = form_data.web.SERPAPI_ENGINE
        request.app.state.config.JINA_API_KEY = form_data.web.JINA_API_KEY
        request.app.state.config.BING_SEARCH_V7_ENDPOINT = (
            form_data.web.BING_SEARCH_V7_ENDPOINT
        )
        request.app.state.config.BING_SEARCH_V7_SUBSCRIPTION_KEY = (
            form_data.web.BING_SEARCH_V7_SUBSCRIPTION_KEY
        )
        request.app.state.config.EXA_API_KEY = form_data.web.EXA_API_KEY
        request.app.state.config.PERPLEXITY_API_KEY = form_data.web.PERPLEXITY_API_KEY
        request.app.state.config.PERPLEXITY_MODEL = form_data.web.PERPLEXITY_MODEL
        request.app.state.config.PERPLEXITY_SEARCH_CONTEXT_USAGE = (
            form_data.web.PERPLEXITY_SEARCH_CONTEXT_USAGE
        )
        request.app.state.config.PERPLEXITY_SEARCH_API_URL = (
            form_data.web.PERPLEXITY_SEARCH_API_URL
        )
        request.app.state.config.SOUGOU_API_SID = form_data.web.SOUGOU_API_SID
        request.app.state.config.SOUGOU_API_SK = form_data.web.SOUGOU_API_SK

        # Web loader settings
        request.app.state.config.WEB_LOADER_ENGINE = form_data.web.WEB_LOADER_ENGINE
        request.app.state.config.WEB_LOADER_TIMEOUT = form_data.web.WEB_LOADER_TIMEOUT

        request.app.state.config.ENABLE_WEB_LOADER_SSL_VERIFICATION = (
            form_data.web.ENABLE_WEB_LOADER_SSL_VERIFICATION
        )
        request.app.state.config.PLAYWRIGHT_WS_URL = form_data.web.PLAYWRIGHT_WS_URL
        request.app.state.config.PLAYWRIGHT_TIMEOUT = form_data.web.PLAYWRIGHT_TIMEOUT
        request.app.state.config.FIRECRAWL_API_KEY = form_data.web.FIRECRAWL_API_KEY
        request.app.state.config.FIRECRAWL_API_BASE_URL = (
            form_data.web.FIRECRAWL_API_BASE_URL
        )
        request.app.state.config.EXTERNAL_WEB_SEARCH_URL = (
            form_data.web.EXTERNAL_WEB_SEARCH_URL
        )
        request.app.state.config.EXTERNAL_WEB_SEARCH_API_KEY = (
            form_data.web.EXTERNAL_WEB_SEARCH_API_KEY
        )
        request.app.state.config.EXTERNAL_WEB_LOADER_URL = (
            form_data.web.EXTERNAL_WEB_LOADER_URL
        )
        request.app.state.config.EXTERNAL_WEB_LOADER_API_KEY = (
            form_data.web.EXTERNAL_WEB_LOADER_API_KEY
        )
        request.app.state.config.TAVILY_EXTRACT_DEPTH = (
            form_data.web.TAVILY_EXTRACT_DEPTH
        )
        request.app.state.config.YOUTUBE_LOADER_LANGUAGE = (
            form_data.web.YOUTUBE_LOADER_LANGUAGE
        )
        request.app.state.config.YOUTUBE_LOADER_PROXY_URL = (
            form_data.web.YOUTUBE_LOADER_PROXY_URL
        )
        request.app.state.YOUTUBE_LOADER_TRANSLATION = (
            form_data.web.YOUTUBE_LOADER_TRANSLATION
        )

    return {
        "status": True,
        # RAG settings
        "RAG_TEMPLATE": request.app.state.config.RAG_TEMPLATE,
        "TOP_K": request.app.state.config.TOP_K,
        "BYPASS_EMBEDDING_AND_RETRIEVAL": request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL,
        "RAG_FULL_CONTEXT": request.app.state.config.RAG_FULL_CONTEXT,
        # Retrieval settings
        "ENABLE_RAG_BM25_SEARCH": request.app.state.config.ENABLE_RAG_BM25_SEARCH,
        "ENABLE_RAG_BM25_ENRICHED_TEXTS": request.app.state.config.ENABLE_RAG_BM25_ENRICHED_TEXTS,
        "ENABLE_RAG_RERANKING": request.app.state.config.ENABLE_RAG_RERANKING,
        "TOP_K_RERANKER": request.app.state.config.TOP_K_RERANKER,
        "RELEVANCE_THRESHOLD": request.app.state.config.RELEVANCE_THRESHOLD,
        "BM25_WEIGHT": request.app.state.config.BM25_WEIGHT,
        # Content extraction settings
        "CONTENT_EXTRACTION_ENGINE": request.app.state.config.CONTENT_EXTRACTION_ENGINE,
        "PDF_EXTRACT_IMAGES": request.app.state.config.PDF_EXTRACT_IMAGES,
        "CONVERSATION_FILE_UPLOAD_EMBEDDING": request.app.state.config.CONVERSATION_FILE_UPLOAD_EMBEDDING,
        "DATALAB_MARKER_API_KEY": request.app.state.config.DATALAB_MARKER_API_KEY,
        "DATALAB_MARKER_API_BASE_URL": request.app.state.config.DATALAB_MARKER_API_BASE_URL,
        "DATALAB_MARKER_ADDITIONAL_CONFIG": request.app.state.config.DATALAB_MARKER_ADDITIONAL_CONFIG,
        "DATALAB_MARKER_SKIP_CACHE": request.app.state.config.DATALAB_MARKER_SKIP_CACHE,
        "DATALAB_MARKER_FORCE_OCR": request.app.state.config.DATALAB_MARKER_FORCE_OCR,
        "DATALAB_MARKER_PAGINATE": request.app.state.config.DATALAB_MARKER_PAGINATE,
        "DATALAB_MARKER_STRIP_EXISTING_OCR": request.app.state.config.DATALAB_MARKER_STRIP_EXISTING_OCR,
        "DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION": request.app.state.config.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION,
        "DATALAB_MARKER_USE_LLM": request.app.state.config.DATALAB_MARKER_USE_LLM,
        "DATALAB_MARKER_OUTPUT_FORMAT": request.app.state.config.DATALAB_MARKER_OUTPUT_FORMAT,
        "EXTERNAL_DOCUMENT_LOADER_URL": request.app.state.config.EXTERNAL_DOCUMENT_LOADER_URL,
        "EXTERNAL_DOCUMENT_LOADER_API_KEY": request.app.state.config.EXTERNAL_DOCUMENT_LOADER_API_KEY,
        "TIKA_SERVER_URL": request.app.state.config.TIKA_SERVER_URL,
        "DOCLING_SERVER_URL": request.app.state.config.DOCLING_SERVER_URL,
        "DOCLING_API_KEY": request.app.state.config.DOCLING_API_KEY,
        "DOCLING_PARAMS": request.app.state.config.DOCLING_PARAMS,
        "DOCUMENT_INTELLIGENCE_ENDPOINT": request.app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT,
        "DOCUMENT_INTELLIGENCE_KEY": request.app.state.config.DOCUMENT_INTELLIGENCE_KEY,
        "DOCUMENT_INTELLIGENCE_MODEL": request.app.state.config.DOCUMENT_INTELLIGENCE_MODEL,
        "MISTRAL_OCR_API_BASE_URL": request.app.state.config.MISTRAL_OCR_API_BASE_URL,
        "MISTRAL_OCR_API_KEY": request.app.state.config.MISTRAL_OCR_API_KEY,
        # MinerU settings
        "MINERU_API_MODE": request.app.state.config.MINERU_API_MODE,
        "MINERU_API_URL": request.app.state.config.MINERU_API_URL,
        "MINERU_API_KEY": request.app.state.config.MINERU_API_KEY,
        "MINERU_API_TIMEOUT": request.app.state.config.MINERU_API_TIMEOUT,
        "MINERU_PARAMS": request.app.state.config.MINERU_PARAMS,
        # Reranking settings
        "RAG_RERANKING_MODEL": request.app.state.config.RAG_RERANKING_MODEL,
        "RAG_RERANKING_ENGINE": request.app.state.config.RAG_RERANKING_ENGINE,
        "RAG_EXTERNAL_RERANKER_URL": request.app.state.config.RAG_EXTERNAL_RERANKER_URL,
        "RAG_EXTERNAL_RERANKER_API_KEY": request.app.state.config.RAG_EXTERNAL_RERANKER_API_KEY,
        "RAG_EXTERNAL_RERANKER_TIMEOUT": request.app.state.config.RAG_EXTERNAL_RERANKER_TIMEOUT,
        "RAG_VOYAGE_RERANKER_URL": request.app.state.config.RAG_VOYAGE_RERANKER_URL,
        "RAG_VOYAGE_RERANKER_API_KEY": request.app.state.config.RAG_VOYAGE_RERANKER_API_KEY,
        "RAG_VOYAGE_RERANKER_TIMEOUT": request.app.state.config.RAG_VOYAGE_RERANKER_TIMEOUT,
        # Chunking settings
        "TEXT_SPLITTER": request.app.state.config.TEXT_SPLITTER,
        "VOYAGE_TOKENIZER_MODEL": request.app.state.config.VOYAGE_TOKENIZER_MODEL,
        "CHUNK_SIZE": request.app.state.config.CHUNK_SIZE,
        "CHUNK_OVERLAP": request.app.state.config.CHUNK_OVERLAP,
        # File upload settings
        "FILE_MAX_SIZE": request.app.state.config.FILE_MAX_SIZE,
        "FILE_MAX_COUNT": request.app.state.config.FILE_MAX_COUNT,
        "FILE_IMAGE_COMPRESSION_WIDTH": request.app.state.config.FILE_IMAGE_COMPRESSION_WIDTH,
        "FILE_IMAGE_COMPRESSION_HEIGHT": request.app.state.config.FILE_IMAGE_COMPRESSION_HEIGHT,
        "ALLOWED_FILE_EXTENSIONS": request.app.state.config.ALLOWED_FILE_EXTENSIONS,
        # Integration settings
        "ENABLE_GOOGLE_DRIVE_INTEGRATION": request.app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION,
        "ENABLE_ONEDRIVE_INTEGRATION": request.app.state.config.ENABLE_ONEDRIVE_INTEGRATION,
        # Web search settings
        "web": {
            "ENABLE_WEB_SEARCH": request.app.state.config.ENABLE_WEB_SEARCH,
            "WEB_SEARCH_ENGINE": request.app.state.config.WEB_SEARCH_ENGINE,
            "WEB_SEARCH_TRUST_ENV": request.app.state.config.WEB_SEARCH_TRUST_ENV,
            "WEB_SEARCH_RESULT_COUNT": request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            "WEB_SEARCH_CONCURRENT_REQUESTS": request.app.state.config.WEB_SEARCH_CONCURRENT_REQUESTS,
            "WEB_LOADER_CONCURRENT_REQUESTS": request.app.state.config.WEB_LOADER_CONCURRENT_REQUESTS,
            "WEB_SEARCH_DOMAIN_FILTER_LIST": request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            "BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL": request.app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL,
            "BYPASS_WEB_SEARCH_WEB_LOADER": request.app.state.config.BYPASS_WEB_SEARCH_WEB_LOADER,
            "SEARXNG_QUERY_URL": request.app.state.config.SEARXNG_QUERY_URL,
            "SEARXNG_LANGUAGE": request.app.state.config.SEARXNG_LANGUAGE,
            "YACY_QUERY_URL": request.app.state.config.YACY_QUERY_URL,
            "YACY_USERNAME": request.app.state.config.YACY_USERNAME,
            "YACY_PASSWORD": request.app.state.config.YACY_PASSWORD,
            "GOOGLE_PSE_API_KEY": request.app.state.config.GOOGLE_PSE_API_KEY,
            "GOOGLE_PSE_ENGINE_ID": request.app.state.config.GOOGLE_PSE_ENGINE_ID,
            "BRAVE_SEARCH_API_KEY": request.app.state.config.BRAVE_SEARCH_API_KEY,
            "KAGI_SEARCH_API_KEY": request.app.state.config.KAGI_SEARCH_API_KEY,
            "MOJEEK_SEARCH_API_KEY": request.app.state.config.MOJEEK_SEARCH_API_KEY,
            "BOCHA_SEARCH_API_KEY": request.app.state.config.BOCHA_SEARCH_API_KEY,
            "SERPSTACK_API_KEY": request.app.state.config.SERPSTACK_API_KEY,
            "SERPSTACK_HTTPS": request.app.state.config.SERPSTACK_HTTPS,
            "SERPER_API_KEY": request.app.state.config.SERPER_API_KEY,
            "SERPLY_API_KEY": request.app.state.config.SERPLY_API_KEY,
            "TAVILY_API_KEY": request.app.state.config.TAVILY_API_KEY,
            "SEARCHAPI_API_KEY": request.app.state.config.SEARCHAPI_API_KEY,
            "SEARCHAPI_ENGINE": request.app.state.config.SEARCHAPI_ENGINE,
            "SERPAPI_API_KEY": request.app.state.config.SERPAPI_API_KEY,
            "SERPAPI_ENGINE": request.app.state.config.SERPAPI_ENGINE,
            "JINA_API_KEY": request.app.state.config.JINA_API_KEY,
            "BING_SEARCH_V7_ENDPOINT": request.app.state.config.BING_SEARCH_V7_ENDPOINT,
            "BING_SEARCH_V7_SUBSCRIPTION_KEY": request.app.state.config.BING_SEARCH_V7_SUBSCRIPTION_KEY,
            "EXA_API_KEY": request.app.state.config.EXA_API_KEY,
            "PERPLEXITY_API_KEY": request.app.state.config.PERPLEXITY_API_KEY,
            "PERPLEXITY_MODEL": request.app.state.config.PERPLEXITY_MODEL,
            "PERPLEXITY_SEARCH_CONTEXT_USAGE": request.app.state.config.PERPLEXITY_SEARCH_CONTEXT_USAGE,
            "PERPLEXITY_SEARCH_API_URL": request.app.state.config.PERPLEXITY_SEARCH_API_URL,
            "SOUGOU_API_SID": request.app.state.config.SOUGOU_API_SID,
            "SOUGOU_API_SK": request.app.state.config.SOUGOU_API_SK,
            "WEB_LOADER_ENGINE": request.app.state.config.WEB_LOADER_ENGINE,
            "WEB_LOADER_TIMEOUT": request.app.state.config.WEB_LOADER_TIMEOUT,
            "ENABLE_WEB_LOADER_SSL_VERIFICATION": request.app.state.config.ENABLE_WEB_LOADER_SSL_VERIFICATION,
            "PLAYWRIGHT_WS_URL": request.app.state.config.PLAYWRIGHT_WS_URL,
            "PLAYWRIGHT_TIMEOUT": request.app.state.config.PLAYWRIGHT_TIMEOUT,
            "FIRECRAWL_API_KEY": request.app.state.config.FIRECRAWL_API_KEY,
            "FIRECRAWL_API_BASE_URL": request.app.state.config.FIRECRAWL_API_BASE_URL,
            "TAVILY_EXTRACT_DEPTH": request.app.state.config.TAVILY_EXTRACT_DEPTH,
            "EXTERNAL_WEB_SEARCH_URL": request.app.state.config.EXTERNAL_WEB_SEARCH_URL,
            "EXTERNAL_WEB_SEARCH_API_KEY": request.app.state.config.EXTERNAL_WEB_SEARCH_API_KEY,
            "EXTERNAL_WEB_LOADER_URL": request.app.state.config.EXTERNAL_WEB_LOADER_URL,
            "EXTERNAL_WEB_LOADER_API_KEY": request.app.state.config.EXTERNAL_WEB_LOADER_API_KEY,
            "YOUTUBE_LOADER_LANGUAGE": request.app.state.config.YOUTUBE_LOADER_LANGUAGE,
            "YOUTUBE_LOADER_PROXY_URL": request.app.state.config.YOUTUBE_LOADER_PROXY_URL,
            "YOUTUBE_LOADER_TRANSLATION": request.app.state.YOUTUBE_LOADER_TRANSLATION,
        },
    }


####################################
#
# Document process and retrieval
#
####################################
class HFTokenTextSplitter:
    def __init__(self, tokenizer, chunk_size: int, chunk_overlap: int, add_start_index: bool = True):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.tokenizer = tokenizer
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.add_start_index = bool(add_start_index)

    def _split_text(self, text: str):
        # 依赖 fast tokenizer 的 offset_mapping，能保证切出来是原文子串（不会 decode 造成不一致）
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        offsets = enc.get("offset_mapping")
        if not offsets:
            # 兜底：没有 offset_mapping 就退化为按字符粗略切（不推荐，但避免直接崩）
            # 你也可以在这里直接 raise
            step = max(1, self.chunk_size - self.chunk_overlap)
            for i in range(0, len(text), step):
                yield (text[i : i + self.chunk_size], i)
            return

        n = len(offsets)
        step = self.chunk_size - self.chunk_overlap

        start_tok = 0
        while start_tok < n:
            end_tok = min(n, start_tok + self.chunk_size)

            # 将 token span 映射回字符 span
            start_char = offsets[start_tok][0]
            end_char = offsets[end_tok - 1][1] if end_tok - 1 >= start_tok else offsets[start_tok][1]

            chunk = text[start_char:end_char]
            yield (chunk, start_char)

            if end_tok == n:
                break
            start_tok += step

    def split_documents(self, docs):
        out = []
        for doc in docs:
            text = doc.page_content
            for chunk, start_idx in self._split_text(text):
                meta = dict(doc.metadata or {})
                if self.add_start_index:
                    meta["start_index"] = start_idx
                out.append(Document(page_content=chunk, metadata=meta))
        return out


class TiktokenTextSplitter:
    def __init__(
        self,
        encoding_name: str,
        chunk_size: int,
        chunk_overlap: int,
        add_start_index: bool = True,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.encoding_name = encoding_name
        self.encoding = get_tiktoken_encoder(encoding_name)
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.add_start_index = bool(add_start_index)

    def split_documents(self, docs):
        out = []
        for doc in docs:
            text = doc.page_content
            tokens = self.encoding.encode(text)
            step = self.chunk_size - self.chunk_overlap
            start_tok = 0
            while start_tok < len(tokens):
                end_tok = min(len(tokens), start_tok + self.chunk_size)
                chunk_tokens = tokens[start_tok:end_tok]
                chunk_text = self.encoding.decode(chunk_tokens)
                meta = dict(doc.metadata or {})
                if self.add_start_index:
                    char_start = len(self.encoding.decode(tokens[:start_tok]))
                    meta["start_index"] = char_start
                out.append(Document(page_content=chunk_text, metadata=meta))
                if end_tok == len(tokens):
                    break
                start_tok += step
        return out


def save_docs_to_vector_db(
    request: Request,
    docs,
    collection_name,
    metadata: Optional[dict] = None,
    overwrite: bool = False,
    split: bool = True,
    add: bool = False,
    user=None,
    effective_config: Optional[dict] = None,
) -> bool:
    def _get_docs_info(docs: list[Document]) -> str:
        docs_info = set()

        # Trying to select relevant metadata identifying the document.
        for doc in docs:
            metadata = getattr(doc, "metadata", {})
            doc_name = metadata.get("name", "")
            if not doc_name:
                doc_name = metadata.get("title", "")
            if not doc_name:
                doc_name = metadata.get("source", "")
            if doc_name:
                docs_info.add(doc_name)

        return ", ".join(docs_info)

    log.debug(
        f"save_docs_to_vector_db: document {_get_docs_info(docs)} {collection_name}"
    )

    # Check if entries with the same hash (metadata.hash) already exist
    if metadata and "hash" in metadata:
        result = VECTOR_DB_CLIENT.query(
            collection_name=collection_name,
            filter={"hash": metadata["hash"]},
        )

        if result is not None:
            existing_doc_ids = result.ids[0]
            if existing_doc_ids:
                log.info(f"Document with hash {metadata['hash']} already exists")
                raise ValueError(ERROR_MESSAGES.DUPLICATE_CONTENT)

    rag_config = effective_config or resolve_collection_rag_config(
        None, request.app.state.config
    )["effective"]

    if split:
        if rag_config["TEXT_SPLITTER"] in ["", "character"]:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=rag_config["CHUNK_SIZE"],
                chunk_overlap=rag_config["CHUNK_OVERLAP"],
                add_start_index=True,
            )
            docs = text_splitter.split_documents(docs)
        elif rag_config["TEXT_SPLITTER"] == "token":
            log.info(
                f"Using token text splitter: tiktoken ({request.app.state.config.TIKTOKEN_ENCODING_NAME})"
            )

            text_splitter = TiktokenTextSplitter(
                encoding_name=request.app.state.config.TIKTOKEN_ENCODING_NAME,
                chunk_size=rag_config["CHUNK_SIZE"],
                chunk_overlap=rag_config["CHUNK_OVERLAP"],
                add_start_index=True,
            )

            docs = text_splitter.split_documents(docs)
        elif rag_config["TEXT_SPLITTER"] == "token_voyage":
            log.info(
                "Using token text splitter: "
                f"{rag_config['VOYAGE_TOKENIZER_MODEL']} (HF AutoTokenizer)"
            )

            text_splitter = HFTokenTextSplitter(
                tokenizer=get_voyage_tokenizer(
                    rag_config["VOYAGE_TOKENIZER_MODEL"]
                ),
                chunk_size=rag_config["CHUNK_SIZE"],
                chunk_overlap=rag_config["CHUNK_OVERLAP"],
                add_start_index=True,
            )

            docs = text_splitter.split_documents(docs)
        else:
            raise ValueError(ERROR_MESSAGES.DEFAULT("Invalid text splitter"))

    if len(docs) == 0:
        raise ValueError(ERROR_MESSAGES.EMPTY_CONTENT)

    texts = [sanitize_text_for_db(doc.page_content) for doc in docs]
    metadatas = [
        {
            **doc.metadata,
            **(metadata if metadata else {}),
            "embedding_config": {
                "engine": rag_config["RAG_EMBEDDING_ENGINE"],
                "model": rag_config["RAG_EMBEDDING_MODEL"],
            },
        }
        for doc in docs
    ]

    try:
        if VECTOR_DB_CLIENT.has_collection(collection_name=collection_name):
            log.info(f"collection {collection_name} already exists")

            if overwrite:
                VECTOR_DB_CLIENT.delete_collection(collection_name=collection_name)
                invalidate_bm25_collections([collection_name])
                log.info(f"deleting existing collection {collection_name}")
            elif add is False:
                log.info(
                    f"collection {collection_name} already exists, overwrite is False and add is False"
                )
                return True

        log.info(f"generating embeddings for {collection_name}")
        embedding_function = get_embedding_function(
            rag_config["RAG_EMBEDDING_ENGINE"],
            rag_config["RAG_EMBEDDING_MODEL"],
            request.app.state.ef,
            (
                request.app.state.config.RAG_OPENAI_API_BASE_URL
                if rag_config["RAG_EMBEDDING_ENGINE"] == "openai"
                else request.app.state.config.RAG_AZURE_OPENAI_BASE_URL
            ),
            (
                request.app.state.config.RAG_OPENAI_API_KEY
                if rag_config["RAG_EMBEDDING_ENGINE"] == "openai"
                else request.app.state.config.RAG_AZURE_OPENAI_API_KEY
            ),
            rag_config["RAG_EMBEDDING_BATCH_SIZE"],
            azure_api_version=(
                request.app.state.config.RAG_AZURE_OPENAI_API_VERSION
                if rag_config["RAG_EMBEDDING_ENGINE"] == "azure_openai"
                else None
            ),
            enable_async=request.app.state.config.ENABLE_ASYNC_EMBEDDING,
        )

        # Run async embedding in sync context
        embeddings = asyncio.run(
            embedding_function(
                list(map(lambda x: x.replace("\n", " "), texts)),
                prefix=RAG_EMBEDDING_CONTENT_PREFIX,
                user=user,
            )
        )
        log.info(f"embeddings generated {len(embeddings)} for {len(texts)} items")

        items = [
            {
                "id": str(uuid.uuid4()),
                "text": text,
                "vector": embeddings[idx],
                "metadata": metadatas[idx],
            }
            for idx, text in enumerate(texts)
        ]

        log.info(f"adding to collection {collection_name}")
        VECTOR_DB_CLIENT.insert(
            collection_name=collection_name,
            items=items,
        )
        mark_bm25_collections_dirty([collection_name])

        log.info(f"added {len(items)} items to collection {collection_name}")
        return True
    except Exception as e:
        log.exception(e)
        raise e


CONVERSATION_UPLOAD_KNOWLEDGE_SYSTEM_TYPE = "conversation_upload"


def is_conversation_upload_processing(form_data: "ProcessFileForm") -> bool:
    # The chat upload pipeline calls /files/ -> /retrieval/process/file without
    # explicit content or target collection.
    return form_data.content is None and form_data.collection_name is None


def get_or_create_user_conversation_upload_knowledge(request: Request, user):
    knowledges = Knowledges.get_knowledge_bases_by_user_id(user.id, permission="write")
    for knowledge in knowledges:
        meta = knowledge.meta or {}
        if (
            meta.get("system_managed") is True
            and meta.get("system_type") == CONVERSATION_UPLOAD_KNOWLEDGE_SYSTEM_TYPE
            and meta.get("owner_user_id") == user.id
        ):
            return knowledge

    owner_name = (getattr(user, "name", "") or "").strip() or user.id
    collection_name = f"{owner_name} Conversation Upload File"
    knowledge = Knowledges.insert_new_knowledge(
        user.id,
        KnowledgeForm(
            name=collection_name,
            description=collection_name,
            access_control={},
            meta={
                "system_managed": True,
                "system_type": CONVERSATION_UPLOAD_KNOWLEDGE_SYSTEM_TYPE,
                "owner_user_id": user.id,
            },
        ),
    )
    if knowledge is None:
        raise ValueError("Failed to create conversation upload knowledge base")

    return knowledge


class ProcessFileForm(BaseModel):
    file_id: str
    content: Optional[str] = None
    collection_name: Optional[str] = None
    knowledge_id: Optional[str] = None


@router.post("/process/file")
def process_file(
    request: Request,
    form_data: ProcessFileForm,
    user=Depends(get_verified_user),
):
    """
    Process a file and save its content to the vector database.
    """
    if user.role == "admin":
        file = Files.get_file_by_id(form_data.file_id)
    else:
        file = Files.get_file_by_id_and_user_id(form_data.file_id, user.id)

    if file:
        try:
            is_conversation_upload = is_conversation_upload_processing(form_data)
            conversation_upload_knowledge = None
            conversation_upload_embedding_enabled = (
                is_conversation_file_upload_embedding_enabled(
                    user=user,
                    global_enabled=request.app.state.config.CONVERSATION_FILE_UPLOAD_EMBEDDING,
                )
            )

            collection_name = form_data.collection_name
            logical_collection_name = form_data.knowledge_id or form_data.collection_name

            if collection_name is None:
                if is_conversation_upload and conversation_upload_embedding_enabled:
                    conversation_upload_knowledge = (
                        get_or_create_user_conversation_upload_knowledge(request, user)
                    )
                    logical_collection_name = conversation_upload_knowledge.id
                    collection_name = get_active_vector_collection_name(
                        conversation_upload_knowledge.id,
                        conversation_upload_knowledge.meta,
                    )
                else:
                    collection_name = f"file-{file.id}"
                    logical_collection_name = None
            else:
                collection_name = get_physical_collection_name(collection_name)

            effective_config = get_collection_effective_config(
                request, logical_collection_name
            )

            if form_data.content:
                # Update the content in the file
                # Usage: /files/{file_id}/data/content/update, /files/ (audio file upload pipeline)

                file_collection_name = f"file-{file.id}"
                try:
                    # /files/{file_id}/data/content/update
                    if VECTOR_DB_CLIENT.has_collection(
                        collection_name=file_collection_name
                    ):
                        VECTOR_DB_CLIENT.delete_collection(
                            collection_name=file_collection_name
                        )
                        invalidate_bm25_collections([file_collection_name])
                    else:
                        log.debug(
                            "Skipping collection delete because collection does not exist: %s",
                            file_collection_name,
                        )
                except Exception as e:
                    # Audio file upload pipeline can race with collection lifecycle.
                    if "does not exist" in str(e).lower():
                        log.debug(
                            "Ignoring missing collection during delete: %s (%s)",
                            file_collection_name,
                            e,
                        )
                    else:
                        raise

                docs = [
                    Document(
                        page_content=form_data.content.replace("<br/>", "\n"),
                        metadata={
                            **file.meta,
                            "name": file.filename,
                            "created_by": file.user_id,
                            "file_id": file.id,
                            "source": file.filename,
                        },
                    )
                ]

                text_content = form_data.content
            elif form_data.collection_name:
                # Check if the file has already been processed and save the content
                # Usage: /knowledge/{id}/file/add, /knowledge/{id}/file/update

                result = VECTOR_DB_CLIENT.query(
                    collection_name=f"file-{file.id}", filter={"file_id": file.id}
                )

                if result is not None and len(result.ids[0]) > 0:
                    docs = [
                        Document(
                            page_content=result.documents[0][idx],
                            metadata=result.metadatas[0][idx],
                        )
                        for idx, id in enumerate(result.ids[0])
                    ]
                else:
                    docs = [
                        Document(
                            page_content=file.data.get("content", ""),
                            metadata={
                                **file.meta,
                                "name": file.filename,
                                "created_by": file.user_id,
                                "file_id": file.id,
                                "source": file.filename,
                            },
                        )
                    ]

                text_content = file.data.get("content", "")
            else:
                # Process the file and save the content
                # Usage: /files/
                file_path = file.path
                if file_path:
                    file_path = Storage.get_file(file_path)
                    loader = Loader(
                        engine=request.app.state.config.CONTENT_EXTRACTION_ENGINE,
                        user=user,
                        DATALAB_MARKER_API_KEY=request.app.state.config.DATALAB_MARKER_API_KEY,
                        DATALAB_MARKER_API_BASE_URL=request.app.state.config.DATALAB_MARKER_API_BASE_URL,
                        DATALAB_MARKER_ADDITIONAL_CONFIG=request.app.state.config.DATALAB_MARKER_ADDITIONAL_CONFIG,
                        DATALAB_MARKER_SKIP_CACHE=request.app.state.config.DATALAB_MARKER_SKIP_CACHE,
                        DATALAB_MARKER_FORCE_OCR=request.app.state.config.DATALAB_MARKER_FORCE_OCR,
                        DATALAB_MARKER_PAGINATE=request.app.state.config.DATALAB_MARKER_PAGINATE,
                        DATALAB_MARKER_STRIP_EXISTING_OCR=request.app.state.config.DATALAB_MARKER_STRIP_EXISTING_OCR,
                        DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION=request.app.state.config.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION,
                        DATALAB_MARKER_FORMAT_LINES=request.app.state.config.DATALAB_MARKER_FORMAT_LINES,
                        DATALAB_MARKER_USE_LLM=request.app.state.config.DATALAB_MARKER_USE_LLM,
                        DATALAB_MARKER_OUTPUT_FORMAT=request.app.state.config.DATALAB_MARKER_OUTPUT_FORMAT,
                        EXTERNAL_DOCUMENT_LOADER_URL=request.app.state.config.EXTERNAL_DOCUMENT_LOADER_URL,
                        EXTERNAL_DOCUMENT_LOADER_API_KEY=request.app.state.config.EXTERNAL_DOCUMENT_LOADER_API_KEY,
                        TIKA_SERVER_URL=request.app.state.config.TIKA_SERVER_URL,
                        DOCLING_SERVER_URL=request.app.state.config.DOCLING_SERVER_URL,
                        DOCLING_API_KEY=request.app.state.config.DOCLING_API_KEY,
                        DOCLING_PARAMS=request.app.state.config.DOCLING_PARAMS,
                        PDF_EXTRACT_IMAGES=request.app.state.config.PDF_EXTRACT_IMAGES,
                        DOCUMENT_INTELLIGENCE_ENDPOINT=request.app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT,
                        DOCUMENT_INTELLIGENCE_KEY=request.app.state.config.DOCUMENT_INTELLIGENCE_KEY,
                        DOCUMENT_INTELLIGENCE_MODEL=request.app.state.config.DOCUMENT_INTELLIGENCE_MODEL,
                        MISTRAL_OCR_API_BASE_URL=request.app.state.config.MISTRAL_OCR_API_BASE_URL,
                        MISTRAL_OCR_API_KEY=request.app.state.config.MISTRAL_OCR_API_KEY,
                        MINERU_API_MODE=request.app.state.config.MINERU_API_MODE,
                        MINERU_API_URL=request.app.state.config.MINERU_API_URL,
                        MINERU_API_KEY=request.app.state.config.MINERU_API_KEY,
                        MINERU_API_TIMEOUT=request.app.state.config.MINERU_API_TIMEOUT,
                        MINERU_PARAMS=request.app.state.config.MINERU_PARAMS,
                    )
                    docs = loader.load(
                        file.filename, file.meta.get("content_type"), file_path
                    )

                    docs = [
                        Document(
                            page_content=doc.page_content,
                            metadata={
                                **filter_metadata(doc.metadata),
                                "name": file.filename,
                                "created_by": file.user_id,
                                "file_id": file.id,
                                "source": file.filename,
                            },
                        )
                        for doc in docs
                    ]
                else:
                    docs = [
                        Document(
                            page_content=file.data.get("content", ""),
                            metadata={
                                **file.meta,
                                "name": file.filename,
                                "created_by": file.user_id,
                                "file_id": file.id,
                                "source": file.filename,
                            },
                        )
                    ]
                text_content = " ".join([doc.page_content for doc in docs])

            log.debug(f"text_content: {text_content}")
            Files.update_file_data_by_id(
                file.id,
                {"content": text_content},
            )
            hash = calculate_sha256_string(text_content)
            Files.update_file_hash_by_id(file.id, hash)

            if is_conversation_upload and not conversation_upload_embedding_enabled:
                Files.update_file_data_by_id(file.id, {"status": "completed"})
                return {
                    "status": True,
                    "collection_name": None,
                    "filename": file.filename,
                    "content": text_content,
                }

            if request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL:
                Files.update_file_data_by_id(file.id, {"status": "completed"})
                return {
                    "status": True,
                    "collection_name": None,
                    "filename": file.filename,
                    "content": text_content,
                }
            else:
                try:
                    if conversation_upload_knowledge:
                        # Reprocessing the same file should overwrite file-scoped vectors.
                        VECTOR_DB_CLIENT.delete(
                            collection_name=collection_name,
                            filter={"file_id": file.id},
                        )
                        invalidate_bm25_collections([collection_name])

                    vector_metadata = {
                        "file_id": file.id,
                        "name": file.filename,
                    }
                    if conversation_upload_knowledge:
                        vector_metadata["source_hash"] = hash
                    else:
                        vector_metadata["hash"] = hash

                    result = save_docs_to_vector_db(
                        request,
                        docs=docs,
                        collection_name=collection_name,
                        metadata=vector_metadata,
                        add=(True if (form_data.collection_name or conversation_upload_knowledge) else False),
                        user=user,
                        effective_config=effective_config,
                    )
                    log.info(f"added {len(docs)} items to collection {collection_name}")

                    if result:
                        file_meta_update = {
                            "collection_name": collection_name,
                        }

                        if conversation_upload_knowledge:
                            Knowledges.add_file_to_knowledge_by_id(
                                knowledge_id=conversation_upload_knowledge.id,
                                file_id=file.id,
                                user_id=user.id,
                            )
                            file_meta_update = {
                                **file_meta_update,
                                "collection_name": conversation_upload_knowledge.id,
                                "active_collection_name": collection_name,
                                "conversation_upload_knowledge_id": conversation_upload_knowledge.id,
                            }

                        Files.update_file_metadata_by_id(file.id, file_meta_update)

                        Files.update_file_data_by_id(
                            file.id,
                            {"status": "completed"},
                        )

                        return {
                            "status": True,
                            "collection_name": collection_name,
                            "filename": file.filename,
                            "content": text_content,
                        }
                    else:
                        raise Exception("Error saving document to vector database")
                except Exception as e:
                    raise e

        except Exception as e:
            log.exception(e)
            Files.update_file_data_by_id(
                file.id,
                {"status": "failed"},
            )

            if "No pandoc was found" in str(e):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED,
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e),
                )

    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )


class ProcessTextForm(BaseModel):
    name: str
    content: str
    collection_name: Optional[str] = None


@router.post("/process/text")
async def process_text(
    request: Request,
    form_data: ProcessTextForm,
    user=Depends(get_verified_user),
):
    collection_name = form_data.collection_name
    if collection_name is None:
        collection_name = calculate_sha256_string(form_data.content)

    docs = [
        Document(
            page_content=form_data.content,
            metadata={"name": form_data.name, "created_by": user.id},
        )
    ]
    text_content = form_data.content
    log.debug(f"text_content: {text_content}")

    result = await run_in_threadpool(
        save_docs_to_vector_db, request, docs, collection_name, user=user
    )
    if result:
        return {
            "status": True,
            "collection_name": collection_name,
            "content": text_content,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(),
        )


@router.post("/process/youtube")
@router.post("/process/web")
async def process_web(
    request: Request,
    form_data: ProcessUrlForm,
    process: bool = Query(True, description="Whether to process and save the content"),
    user=Depends(get_verified_user),
):
    try:
        content, docs = await run_in_threadpool(
            get_content_from_url, request, form_data.url
        )
        log.debug(f"text_content: {content}")

        if process:
            collection_name = form_data.collection_name
            if not collection_name:
                collection_name = calculate_sha256_string(form_data.url)[:63]

            if not request.app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL:
                await run_in_threadpool(
                    save_docs_to_vector_db,
                    request,
                    docs,
                    collection_name,
                    overwrite=True,
                    user=user,
                )
            else:
                collection_name = None

            return {
                "status": True,
                "collection_name": collection_name,
                "filename": form_data.url,
                "file": {
                    "data": {
                        "content": content,
                    },
                    "meta": {
                        "name": form_data.url,
                        "source": form_data.url,
                    },
                },
            }
        else:
            return {
                "status": True,
                "content": content,
            }
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


def search_web(
    request: Request, engine: str, query: str, user=None
) -> list[SearchResult]:
    """Search the web using a search engine and return the results as a list of SearchResult objects.
    Will look for a search engine API key in environment variables in the following order:
    - SEARXNG_QUERY_URL
    - YACY_QUERY_URL + YACY_USERNAME + YACY_PASSWORD
    - GOOGLE_PSE_API_KEY + GOOGLE_PSE_ENGINE_ID
    - BRAVE_SEARCH_API_KEY
    - KAGI_SEARCH_API_KEY
    - MOJEEK_SEARCH_API_KEY
    - BOCHA_SEARCH_API_KEY
    - SERPSTACK_API_KEY
    - SERPER_API_KEY
    - SERPLY_API_KEY
    - TAVILY_API_KEY
    - EXA_API_KEY
    - PERPLEXITY_API_KEY
    - SOUGOU_API_SID + SOUGOU_API_SK
    - SEARCHAPI_API_KEY + SEARCHAPI_ENGINE (by default `google`)
    - SERPAPI_API_KEY + SERPAPI_ENGINE (by default `google`)
    Args:
        query (str): The query to search for
    """

    # TODO: add playwright to search the web
    if engine == "perplexity_search":
        if request.app.state.config.PERPLEXITY_API_KEY:
            return search_perplexity_search(
                request.app.state.config.PERPLEXITY_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
                request.app.state.config.PERPLEXITY_SEARCH_API_URL,
                user,
            )
        else:
            raise Exception("No PERPLEXITY_API_KEY found in environment variables")
    elif engine == "searxng":
        if request.app.state.config.SEARXNG_QUERY_URL:
            searxng_kwargs = {"language": request.app.state.config.SEARXNG_LANGUAGE}
            return search_searxng(
                request.app.state.config.SEARXNG_QUERY_URL,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
                **searxng_kwargs,
            )
        else:
            raise Exception("No SEARXNG_QUERY_URL found in environment variables")
    elif engine == "yacy":
        if request.app.state.config.YACY_QUERY_URL:
            return search_yacy(
                request.app.state.config.YACY_QUERY_URL,
                request.app.state.config.YACY_USERNAME,
                request.app.state.config.YACY_PASSWORD,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No YACY_QUERY_URL found in environment variables")
    elif engine == "google_pse":
        if (
            request.app.state.config.GOOGLE_PSE_API_KEY
            and request.app.state.config.GOOGLE_PSE_ENGINE_ID
        ):
            return search_google_pse(
                request.app.state.config.GOOGLE_PSE_API_KEY,
                request.app.state.config.GOOGLE_PSE_ENGINE_ID,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
                referer=request.app.state.config.WEBUI_URL,
            )
        else:
            raise Exception(
                "No GOOGLE_PSE_API_KEY or GOOGLE_PSE_ENGINE_ID found in environment variables"
            )
    elif engine == "brave":
        if request.app.state.config.BRAVE_SEARCH_API_KEY:
            return search_brave(
                request.app.state.config.BRAVE_SEARCH_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No BRAVE_SEARCH_API_KEY found in environment variables")
    elif engine == "kagi":
        if request.app.state.config.KAGI_SEARCH_API_KEY:
            return search_kagi(
                request.app.state.config.KAGI_SEARCH_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No KAGI_SEARCH_API_KEY found in environment variables")
    elif engine == "mojeek":
        if request.app.state.config.MOJEEK_SEARCH_API_KEY:
            return search_mojeek(
                request.app.state.config.MOJEEK_SEARCH_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No MOJEEK_SEARCH_API_KEY found in environment variables")
    elif engine == "bocha":
        if request.app.state.config.BOCHA_SEARCH_API_KEY:
            return search_bocha(
                request.app.state.config.BOCHA_SEARCH_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No BOCHA_SEARCH_API_KEY found in environment variables")
    elif engine == "serpstack":
        if request.app.state.config.SERPSTACK_API_KEY:
            return search_serpstack(
                request.app.state.config.SERPSTACK_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
                https_enabled=request.app.state.config.SERPSTACK_HTTPS,
            )
        else:
            raise Exception("No SERPSTACK_API_KEY found in environment variables")
    elif engine == "serper":
        if request.app.state.config.SERPER_API_KEY:
            return search_serper(
                request.app.state.config.SERPER_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No SERPER_API_KEY found in environment variables")
    elif engine == "serply":
        if request.app.state.config.SERPLY_API_KEY:
            return search_serply(
                request.app.state.config.SERPLY_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                filter_list=request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No SERPLY_API_KEY found in environment variables")
    elif engine == "duckduckgo":
        return search_duckduckgo(
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            concurrent_requests=request.app.state.config.WEB_SEARCH_CONCURRENT_REQUESTS,
        )
    elif engine == "tavily":
        if request.app.state.config.TAVILY_API_KEY:
            return search_tavily(
                request.app.state.config.TAVILY_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No TAVILY_API_KEY found in environment variables")
    elif engine == "exa":
        if request.app.state.config.EXA_API_KEY:
            return search_exa(
                request.app.state.config.EXA_API_KEY,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No EXA_API_KEY found in environment variables")
    elif engine == "searchapi":
        if request.app.state.config.SEARCHAPI_API_KEY:
            return search_searchapi(
                request.app.state.config.SEARCHAPI_API_KEY,
                request.app.state.config.SEARCHAPI_ENGINE,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No SEARCHAPI_API_KEY found in environment variables")
    elif engine == "serpapi":
        if request.app.state.config.SERPAPI_API_KEY:
            return search_serpapi(
                request.app.state.config.SERPAPI_API_KEY,
                request.app.state.config.SERPAPI_ENGINE,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception("No SERPAPI_API_KEY found in environment variables")
    elif engine == "jina":
        return search_jina(
            request.app.state.config.JINA_API_KEY,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
        )
    elif engine == "bing":
        return search_bing(
            request.app.state.config.BING_SEARCH_V7_SUBSCRIPTION_KEY,
            request.app.state.config.BING_SEARCH_V7_ENDPOINT,
            str(DEFAULT_LOCALE),
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
        )
    elif engine == "azure":
        if (
            request.app.state.config.AZURE_AI_SEARCH_API_KEY
            and request.app.state.config.AZURE_AI_SEARCH_ENDPOINT
            and request.app.state.config.AZURE_AI_SEARCH_INDEX_NAME
        ):
            return search_azure(
                request.app.state.config.AZURE_AI_SEARCH_API_KEY,
                request.app.state.config.AZURE_AI_SEARCH_ENDPOINT,
                request.app.state.config.AZURE_AI_SEARCH_INDEX_NAME,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception(
                "AZURE_AI_SEARCH_API_KEY, AZURE_AI_SEARCH_ENDPOINT, and AZURE_AI_SEARCH_INDEX_NAME are required for Azure AI Search"
            )
    elif engine == "exa":
        return search_exa(
            request.app.state.config.EXA_API_KEY,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
        )
    elif engine == "perplexity":
        return search_perplexity(
            request.app.state.config.PERPLEXITY_API_KEY,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            model=request.app.state.config.PERPLEXITY_MODEL,
            search_context_usage=request.app.state.config.PERPLEXITY_SEARCH_CONTEXT_USAGE,
        )
    elif engine == "sougou":
        if (
            request.app.state.config.SOUGOU_API_SID
            and request.app.state.config.SOUGOU_API_SK
        ):
            return search_sougou(
                request.app.state.config.SOUGOU_API_SID,
                request.app.state.config.SOUGOU_API_SK,
                query,
                request.app.state.config.WEB_SEARCH_RESULT_COUNT,
                request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            )
        else:
            raise Exception(
                "No SOUGOU_API_SID or SOUGOU_API_SK found in environment variables"
            )
    elif engine == "firecrawl":
        return search_firecrawl(
            request.app.state.config.FIRECRAWL_API_BASE_URL,
            request.app.state.config.FIRECRAWL_API_KEY,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
        )
    elif engine == "external":
        return search_external(
            request,
            request.app.state.config.EXTERNAL_WEB_SEARCH_URL,
            request.app.state.config.EXTERNAL_WEB_SEARCH_API_KEY,
            query,
            request.app.state.config.WEB_SEARCH_RESULT_COUNT,
            request.app.state.config.WEB_SEARCH_DOMAIN_FILTER_LIST,
            user=user,
        )
    else:
        raise Exception("No search engine API key found in environment variables")


@router.post("/process/web/search")
async def process_web_search(
    request: Request, form_data: SearchForm, user=Depends(get_verified_user)
):

    urls = []
    result_items = []

    try:
        logging.debug(
            f"trying to web search with {request.app.state.config.WEB_SEARCH_ENGINE, form_data.queries}"
        )

        # Use semaphore to limit concurrent requests based on WEB_SEARCH_CONCURRENT_REQUESTS
        # 0 or None = unlimited (previous behavior), positive number = limited concurrency
        # Set to 1 for sequential execution (rate-limited APIs like Brave free tier)
        concurrent_limit = request.app.state.config.WEB_SEARCH_CONCURRENT_REQUESTS

        if concurrent_limit:
            # Limited concurrency with semaphore
            semaphore = asyncio.Semaphore(concurrent_limit)

            async def search_with_limit(query):
                async with semaphore:
                    return await run_in_threadpool(
                        search_web,
                        request,
                        request.app.state.config.WEB_SEARCH_ENGINE,
                        query,
                        user,
                    )

            search_tasks = [search_with_limit(query) for query in form_data.queries]
        else:
            # Unlimited parallel execution (previous behavior)
            search_tasks = [
                run_in_threadpool(
                    search_web,
                    request,
                    request.app.state.config.WEB_SEARCH_ENGINE,
                    query,
                    user,
                )
                for query in form_data.queries
            ]

        search_results = await asyncio.gather(*search_tasks)

        for result in search_results:
            if result:
                for item in result:
                    if item and item.link:
                        result_items.append(item)
                        urls.append(item.link)

        urls = list(dict.fromkeys(urls))
        log.debug(f"urls: {urls}")

    except Exception as e:
        log.exception(e)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.WEB_SEARCH_ERROR(e),
        )

    if len(urls) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.DEFAULT("No results found from web search"),
        )

    try:
        if request.app.state.config.BYPASS_WEB_SEARCH_WEB_LOADER:
            search_results = [
                item for result in search_results for item in result if result
            ]

            docs = [
                Document(
                    page_content=result.snippet,
                    metadata={
                        "source": result.link,
                        "title": result.title,
                        "snippet": result.snippet,
                        "link": result.link,
                    },
                )
                for result in search_results
                if hasattr(result, "snippet") and result.snippet is not None
            ]
        else:
            loader = get_web_loader(
                urls,
                verify_ssl=request.app.state.config.ENABLE_WEB_LOADER_SSL_VERIFICATION,
                requests_per_second=request.app.state.config.WEB_LOADER_CONCURRENT_REQUESTS,
                trust_env=request.app.state.config.WEB_SEARCH_TRUST_ENV,
            )
            docs = await loader.aload()

        urls = [
            doc.metadata.get("source") for doc in docs if doc.metadata.get("source")
        ]  # only keep the urls returned by the loader
        result_items = [
            dict(item) for item in result_items if item.link in urls
        ]  # only keep the search results that have been loaded

        if request.app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL:
            return {
                "status": True,
                "collection_name": None,
                "filenames": urls,
                "items": result_items,
                "docs": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in docs
                ],
                "loaded_count": len(docs),
            }
        else:
            # Create a single collection for all documents
            collection_name = (
                f"web-search-{calculate_sha256_string('-'.join(form_data.queries))}"[
                    :63
                ]
            )

            try:
                await run_in_threadpool(
                    save_docs_to_vector_db,
                    request,
                    docs,
                    collection_name,
                    overwrite=True,
                    user=user,
                )
            except Exception as e:
                log.debug(f"error saving docs: {e}")

            return {
                "status": True,
                "collection_names": [collection_name],
                "items": result_items,
                "filenames": urls,
                "loaded_count": len(docs),
            }
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


class QueryDocForm(BaseModel):
    collection_name: str
    query: str
    k: Optional[int] = None
    k_reranker: Optional[int] = None
    r: Optional[float] = None
    enable_bm25_search: Optional[bool] = None
    enable_reranking: Optional[bool] = None
    bm25_weight: Optional[float] = None
    enable_bm25_enriched_texts: Optional[bool] = None


@router.post("/query/doc")
async def query_doc_handler(
    request: Request,
    form_data: QueryDocForm,
    user=Depends(get_verified_user),
):
    try:
        physical_collection_name = get_physical_collection_name(form_data.collection_name)
        effective_config = get_collection_effective_config(
            request, form_data.collection_name
        )
        embedding_function = build_embedding_function_from_effective_config(
            request, effective_config
        )

        enable_bm25_search = (
            form_data.enable_bm25_search
            if form_data.enable_bm25_search is not None
            else request.app.state.config.ENABLE_RAG_BM25_SEARCH
        )
        enable_reranking = (
            form_data.enable_reranking
            if form_data.enable_reranking is not None
            else request.app.state.config.ENABLE_RAG_RERANKING
        )
        reranking_function = (
            build_reranking_function_from_effective_config(request, effective_config)
            if enable_reranking
            else None
        )

        collection_result = None
        if enable_bm25_search:
            collection_result = VECTOR_DB_CLIENT.get(
                collection_name=physical_collection_name
            )

        return await query_doc_with_rag_pipeline(
            collection_name=physical_collection_name,
            collection_result=collection_result,
            query=form_data.query,
            embedding_function=lambda query, prefix: embedding_function(
                query, prefix=prefix, user=user
            ),
            k=form_data.k if form_data.k else effective_config["TOP_K"],
            reranking_function=(
                (
                    lambda query, documents: reranking_function(
                        query, documents, user=user
                    )
                )
                if reranking_function
                else None
            ),
            k_reranker=form_data.k_reranker or effective_config["TOP_K_RERANKER"],
            r=(
                form_data.r
                if form_data.r
                else effective_config["RELEVANCE_THRESHOLD"]
            ),
            bm25_weight=(
                form_data.bm25_weight
                if form_data.bm25_weight is not None
                else request.app.state.config.BM25_WEIGHT
            ),
            enable_bm25_search=enable_bm25_search,
            enable_reranking=enable_reranking,
            enable_bm25_enriched_texts=(
                form_data.enable_bm25_enriched_texts
                if form_data.enable_bm25_enriched_texts is not None
                else request.app.state.config.ENABLE_RAG_BM25_ENRICHED_TEXTS
            ),
        )
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


class QueryCollectionsForm(BaseModel):
    collection_names: list[str]
    query: str
    k: Optional[int] = None
    k_reranker: Optional[int] = None
    r: Optional[float] = None
    enable_bm25_search: Optional[bool] = None
    enable_reranking: Optional[bool] = None
    bm25_weight: Optional[float] = None
    enable_bm25_enriched_texts: Optional[bool] = None


@router.post("/query/collection")
async def query_collection_handler(
    request: Request,
    form_data: QueryCollectionsForm,
    user=Depends(get_verified_user),
):
    try:
        effective_base_k = form_data.k if form_data.k else request.app.state.config.TOP_K
        enable_bm25_search = (
            form_data.enable_bm25_search
            if form_data.enable_bm25_search is not None
            else request.app.state.config.ENABLE_RAG_BM25_SEARCH
        )
        enable_reranking = (
            form_data.enable_reranking
            if form_data.enable_reranking is not None
            else request.app.state.config.ENABLE_RAG_RERANKING
        )
        results = []

        for logical_collection_name in form_data.collection_names:
            physical_collection_name = get_physical_collection_name(logical_collection_name)
            effective_config = get_collection_effective_config(
                request, logical_collection_name
            )

            embedding_function = build_embedding_function_from_effective_config(
                request, effective_config
            )
            reranking_function = (
                build_reranking_function_from_effective_config(request, effective_config)
                if enable_reranking
                else None
            )

            collection_result = None
            if enable_bm25_search:
                collection_result = VECTOR_DB_CLIENT.get(
                    collection_name=physical_collection_name
                )

            query_result = await query_doc_with_rag_pipeline(
                collection_name=physical_collection_name,
                collection_result=collection_result,
                query=form_data.query,
                embedding_function=lambda query, prefix: embedding_function(
                    query, prefix=prefix, user=user
                ),
                k=effective_base_k,
                reranking_function=(
                    (
                        lambda query, documents: reranking_function(
                            query, documents, user=user
                        )
                    )
                    if reranking_function
                    else None
                ),
                k_reranker=(
                    form_data.k_reranker
                    if form_data.k_reranker is not None
                    else effective_config["TOP_K_RERANKER"]
                ),
                r=(
                    form_data.r
                    if form_data.r is not None
                    else effective_config["RELEVANCE_THRESHOLD"]
                ),
                bm25_weight=(
                    form_data.bm25_weight
                    if form_data.bm25_weight is not None
                    else request.app.state.config.BM25_WEIGHT
                ),
                enable_bm25_search=enable_bm25_search,
                enable_reranking=enable_reranking,
                enable_bm25_enriched_texts=(
                    form_data.enable_bm25_enriched_texts
                    if form_data.enable_bm25_enriched_texts is not None
                    else request.app.state.config.ENABLE_RAG_BM25_ENRICHED_TEXTS
                ),
            )
            results.append(query_result)

        return merge_and_sort_query_results(results, k=effective_base_k)

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )



####################################
#
# Vector DB operations
#
####################################


class DeleteForm(BaseModel):
    collection_name: str
    file_id: str


@router.post("/delete")
def delete_entries_from_collection(form_data: DeleteForm, user=Depends(get_admin_user)):
    try:
        if VECTOR_DB_CLIENT.has_collection(collection_name=form_data.collection_name):
            file = Files.get_file_by_id(form_data.file_id)
            hash = file.hash

            VECTOR_DB_CLIENT.delete(
                collection_name=form_data.collection_name,
                metadata={"hash": hash},
            )
            mark_bm25_collections_dirty([form_data.collection_name])
            return {"status": True}
        else:
            return {"status": False}
    except Exception as e:
        log.exception(e)
        return {"status": False}


@router.post("/reset/db")
def reset_vector_db(user=Depends(get_admin_user)):
    VECTOR_DB_CLIENT.reset()
    clear_bm25_index_cache()
    Knowledges.delete_all_knowledge()


@router.post("/reset/uploads")
def reset_upload_dir(user=Depends(get_admin_user)) -> bool:
    folder = f"{UPLOAD_DIR}"
    try:
        # Check if the directory exists
        if os.path.exists(folder):
            # Iterate over all the files and directories in the specified directory
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove the file or link
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove the directory
                except Exception as e:
                    log.exception(f"Failed to delete {file_path}. Reason: {e}")
        else:
            log.warning(f"The directory {folder} does not exist")
    except Exception as e:
        log.exception(f"Failed to process the directory {folder}. Reason: {e}")
    return True


@router.post("/reset/knowledge-uploads")
def reset_knowledge_upload_dir(user=Depends(get_admin_user)) -> bool:
    folder = f"{KNOWLEDGE_UPLOAD_DIR}"
    try:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    log.exception(f"Failed to delete {file_path}. Reason: {e}")
        else:
            log.warning(f"The directory {folder} does not exist")
    except Exception as e:
        log.exception(f"Failed to process the directory {folder}. Reason: {e}")
    return True


if ENV == "dev":

    @router.get("/ef/{text}")
    async def get_embeddings(request: Request, text: Optional[str] = "Hello World!"):
        return {
            "result": await request.app.state.EMBEDDING_FUNCTION(
                text, prefix=RAG_EMBEDDING_QUERY_PREFIX
            )
        }


class BatchProcessFilesForm(BaseModel):
    files: List[FileModel]
    collection_name: str


class BatchProcessFilesResult(BaseModel):
    file_id: str
    status: str
    error: Optional[str] = None


class BatchProcessFilesResponse(BaseModel):
    results: List[BatchProcessFilesResult]
    errors: List[BatchProcessFilesResult]


@router.post("/process/files/batch")
async def process_files_batch(
    request: Request,
    form_data: BatchProcessFilesForm,
    user=Depends(get_verified_user),
) -> BatchProcessFilesResponse:
    """
    Process a batch of files and save them to the vector database.
    """

    collection_name = form_data.collection_name

    file_results: List[BatchProcessFilesResult] = []
    file_errors: List[BatchProcessFilesResult] = []
    file_updates: List[FileUpdateForm] = []

    # Prepare all documents first
    all_docs: List[Document] = []

    for file in form_data.files:
        try:
            text_content = file.data.get("content", "")
            docs: List[Document] = [
                Document(
                    page_content=text_content.replace("<br/>", "\n"),
                    metadata={
                        **file.meta,
                        "name": file.filename,
                        "created_by": file.user_id,
                        "file_id": file.id,
                        "source": file.filename,
                    },
                )
            ]

            all_docs.extend(docs)

            file_updates.append(
                FileUpdateForm(
                    hash=calculate_sha256_string(text_content),
                    data={"content": text_content},
                )
            )
            file_results.append(
                BatchProcessFilesResult(file_id=file.id, status="prepared")
            )

        except Exception as e:
            log.error(f"process_files_batch: Error processing file {file.id}: {str(e)}")
            file_errors.append(
                BatchProcessFilesResult(file_id=file.id, status="failed", error=str(e))
            )

    # Save all documents in one batch
    if all_docs:
        try:
            await run_in_threadpool(
                save_docs_to_vector_db,
                request,
                all_docs,
                collection_name,
                add=True,
                user=user,
            )

            # Update all files with collection name
            for file_update, file_result in zip(file_updates, file_results):
                Files.update_file_by_id(id=file_result.file_id, form_data=file_update)
                file_result.status = "completed"

        except Exception as e:
            log.error(
                f"process_files_batch: Error saving documents to vector DB: {str(e)}"
            )
            for file_result in file_results:
                file_result.status = "failed"
                file_errors.append(
                    BatchProcessFilesResult(file_id=file_result.file_id, error=str(e))
                )

    return BatchProcessFilesResponse(results=file_results, errors=file_errors)
