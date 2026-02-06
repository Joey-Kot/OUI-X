# syntax=docker/dockerfile:1
# Initialize device type args
# use build args in the docker build command with --build-arg="BUILDARG=true"
ARG USE_CUDA=false
ARG USE_OLLAMA=false
ARG USE_SLIM=false
ARG USE_PERMISSION_HARDENING=false
# Tested with cu117 for CUDA 11 and cu121 for CUDA 12 (default)
ARG USE_CUDA_VER=cu128
# any sentence transformer model; models to use can be found at https://huggingface.co/models?library=sentence-transformers
# Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
# for better performance and multilangauge support use "intfloat/multilingual-e5-large" (~2.5GB) or "intfloat/multilingual-e5-base" (~1.5GB)
# IMPORTANT: If you change the embedding model (sentence-transformers/all-MiniLM-L6-v2) and vice versa, you aren't able to use RAG Chat with your previous documents loaded in the WebUI! You need to re-embed them.
# ARG USE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG USE_EMBEDDING_MODEL=""
ARG USE_RERANKING_MODEL=""

# Tiktoken encoding name; models to use can be found at https://huggingface.co/models?library=tiktoken
# ARG USE_TIKTOKEN_ENCODING_NAME="cl100k_base"
ARG USE_TIKTOKEN_ENCODING_NAME=""

ARG BUILD_HASH=dev-build
# Override at your own risk - non-root configurations are untested
ARG UID=0
ARG GID=0

######## WebUI frontend ########
FROM --platform=$BUILDPLATFORM node:22-alpine3.20 AS build
ARG BUILD_HASH

# Set Node.js options (heap limit Allocation failed - JavaScript heap out of memory)
# ENV NODE_OPTIONS="--max-old-space-size=4096"

WORKDIR /app

# to store git revision in build
RUN apk add --no-cache git

COPY package.json package-lock.json ./
RUN npm ci --force

COPY . .
ENV APP_BUILD_HASH=${BUILD_HASH}
RUN npm run build

######## WebUI backend ########
FROM python:3.11-slim-bookworm AS base

# Use args
# ARG USE_CUDA
# ARG USE_OLLAMA
# ARG USE_CUDA_VER
# ARG USE_SLIM
ARG USE_PERMISSION_HARDENING
# ARG USE_EMBEDDING_MODEL
# ARG USE_RERANKING_MODEL
ARG UID
ARG GID

# Python settings
ENV PYTHONUNBUFFERED=1

## Basis ##
ENV ENV=prod \
    PORT=8080
    # pass build args to the build
    # USE_OLLAMA_DOCKER=${USE_OLLAMA} \
    # USE_CUDA_DOCKER=${USE_CUDA} \
    # USE_SLIM_DOCKER=${USE_SLIM} \
    # USE_CUDA_DOCKER_VER=${USE_CUDA_VER} \
    # USE_EMBEDDING_MODEL_DOCKER=${USE_EMBEDDING_MODEL} \
    # USE_RERANKING_MODEL_DOCKER=${USE_RERANKING_MODEL}

## Basis URL Config ##
# ENV OLLAMA_BASE_URL="/ollama" \
#     OPENAI_API_BASE_URL=""

## API Key and Security Config ##
ENV OPENAI_API_KEY="" \
    WEBUI_SECRET_KEY="" \
    SCARF_NO_ANALYTICS=true \
    DO_NOT_TRACK=true \
    ANONYMIZED_TELEMETRY=false

#### Other models #########################################################
## whisper TTS model settings ##
# ENV WHISPER_MODEL="base" \
#     WHISPER_MODEL_DIR="/app/backend/data/cache/whisper/models"

## RAG Embedding model settings ##
# ENV RAG_EMBEDDING_MODEL="$USE_EMBEDDING_MODEL_DOCKER" \
#     RAG_RERANKING_MODEL="$USE_RERANKING_MODEL_DOCKER" \
#     SENTENCE_TRANSFORMERS_HOME="/app/backend/data/cache/embedding/models"

## Tiktoken model settings ##
# ENV TIKTOKEN_ENCODING_NAME="cl100k_base" \
#     TIKTOKEN_CACHE_DIR="/app/backend/data/cache/tiktoken"

## Hugging Face download cache ##
# ENV HF_HOME="/app/backend/data/cache/embedding/models"

## Torch Extensions ##
# ENV TORCH_EXTENSIONS_DIR="/.cache/torch_extensions"

#### Other models ##########################################################

WORKDIR /app/backend

ENV HOME=/root
# Create user and group if not root
RUN if [ $UID -ne 0 ]; then \
    if [ $GID -ne 0 ]; then \
    addgroup --gid $GID app; \
    fi; \
    adduser --uid $UID --gid $GID --home $HOME --disabled-password --no-create-home app; \
    fi

RUN mkdir -p $HOME/.cache/chroma
RUN echo -n 00000000-0000-0000-0000-000000000000 > $HOME/.cache/chroma/telemetry_user_id

# Make sure the user has access to the app and root directory
RUN chown -R $UID:$GID /app $HOME

# Install common system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git build-essential pandoc gcc netcat-openbsd curl jq \
    python3-dev \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# install python dependencies
# COPY --chown=$UID:$GID ./backend/requirements.txt ./requirements.txt
COPY --chown=$UID:$GID ./backend/requirements-min.txt ./requirements.txt

RUN pip3 install --no-cache-dir uv && \
    # if [ "$USE_CUDA" = "true" ]; then \
    # # If you use CUDA the whisper and embedding model will be downloaded on first use
    # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$USE_CUDA_DOCKER_VER --no-cache-dir && \
    uv pip install --system -r requirements.txt --no-cache-dir && \
    # python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['RAG_EMBEDDING_MODEL'], device='cpu')" && \
    # python -c "import os; from faster_whisper import WhisperModel; WhisperModel(os.environ['WHISPER_MODEL'], device='cpu', compute_type='int8', download_root=os.environ['WHISPER_MODEL_DIR'])"; \
    # python -c "import os; import tiktoken; tiktoken.get_encoding(os.environ['TIKTOKEN_ENCODING_NAME'])"; \
    # else \
    # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir && \
    # uv pip install --system -r requirements.txt --no-cache-dir && \
    # if [ "$USE_SLIM" != "true" ]; then \
    # python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['RAG_EMBEDDING_MODEL'], device='cpu')" && \
    # python -c "import os; from faster_whisper import WhisperModel; WhisperModel(os.environ['WHISPER_MODEL'], device='cpu', compute_type='int8', download_root=os.environ['WHISPER_MODEL_DIR'])"; \
    # python -c "import os; import tiktoken; tiktoken.get_encoding(os.environ['TIKTOKEN_ENCODING_NAME'])"; \
    # fi; \
    # fi; \
    mkdir -p /app/backend/data && chown -R $UID:$GID /app/backend/data/ && \
    rm -rf /var/lib/apt/lists/*;

# 补上面批量注释导致缺失的依赖
RUN uv pip install --system \
    authlib==1.6.6 \
    python-mimeparse==2.0.0 \
    tiktoken \
    anthropic \
    google-genai==1.56.0 \
    google-generativeai==0.8.6 \
    langchain==1.2.0 \
    langchain-community==0.4.1 \
    langchain-classic==1.0.0 \
    langchain-text-splitters==1.1.0 \
    weaviate-client==4.19.0 \
    opensearch-py==3.1.0 \
    transformers==4.57.3 \
    # sentence-transformers==5.2.0 \
    # accelerate \
    pyarrow==20.0.0 \
    # einops==0.8.1  \
    ftfy==6.3.1 \
    chardet==5.2.0 \
    pypdf==6.5.0 \
    fpdf2==2.8.5 \
    pymdown-extensions==10.19.1 \
    docx2txt==0.9 \
    python-pptx==1.0.2 \
    unstructured==0.18.21 \
    msoffcrypto-tool==5.4.2 \
    nltk==3.9.2 \
    Markdown==3.10 \
    pypandoc==1.16.2 \
    pandas==2.3.3 \
    openpyxl==3.1.5 \
    pyxlsb==1.0.10 \
    xlrd==2.0.2 \
    validators==0.35.0 \
    psutil \
    # sentencepiece \
    soundfile==0.13.1 \
    pillow==12.0.0 \
    # opencv-python-headless==4.12.0.88 \
    # rapidocr-onnxruntime==1.4.4 \
    rank-bm25==0.2.2 \
    # onnxruntime==1.23.2 \
    # faster-whisper==1.2.1 \
    youtube-transcript-api==1.2.3 \
    pytube==15.0.0 \
    ddgs==9.10.0 \
    azure-ai-documentintelligence==1.0.2 \
    azure-identity==1.25.1 \
    azure-storage-blob==12.27.1 \
    azure-search-documents==11.6.0 \
    google-api-python-client \
    google-auth-httplib2 \
    google-auth-oauthlib \
    googleapis-common-protos==1.72.0 \
    google-cloud-storage==3.7.0 \
    pymongo \
    psycopg2-binary==2.9.11 \
    pgvector==0.4.2 \
    PyMySQL==1.1.2 \
    boto3==1.42.14 \
    pymilvus==2.6.5 \
    qdrant-client==1.16.2 \
    playwright==1.57.0 \
    elasticsearch==9.2.0 \
    pinecone==6.0.2 \
    oracledb==3.4.1 \
    # av==14.0.1 \
    colbert-ai==0.2.22 \
    docker~=7.1.0 \
    pytest~=8.4.1 \
    pytest-docker~=3.2.5 \
    ldap3==2.9.1 \
    firecrawl-py==4.12.0 \
    opentelemetry-api==1.39.1 \
    opentelemetry-sdk==1.39.1 \
    opentelemetry-exporter-otlp==1.39.1 \
    opentelemetry-instrumentation==0.60b1 \
    opentelemetry-instrumentation-fastapi==0.60b1 \
    opentelemetry-instrumentation-sqlalchemy==0.60b1 \
    opentelemetry-instrumentation-redis==0.60b1 \
    opentelemetry-instrumentation-requests==0.60b1 \
    opentelemetry-instrumentation-logging==0.60b1 \
    opentelemetry-instrumentation-httpx==0.60b1 \
    opentelemetry-instrumentation-aiohttp-client==0.60b1 \
    --no-cache-dir

# Install Ollama if requested
# RUN if [ "$USE_OLLAMA" = "true" ]; then \
#     date +%s > /tmp/ollama_build_hash && \
#     echo "Cache broken at timestamp: `cat /tmp/ollama_build_hash`" && \
#     curl -fsSL https://ollama.com/install.sh | sh && \
#     rm -rf /var/lib/apt/lists/*; \
#     fi

# copy embedding weight from build
# RUN mkdir -p /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2
# COPY --from=build /app/onnx /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx

# copy built frontend files
COPY --chown=$UID:$GID --from=build /app/build /app/build
COPY --chown=$UID:$GID --from=build /app/CHANGELOG.md /app/CHANGELOG.md
COPY --chown=$UID:$GID --from=build /app/package.json /app/package.json

# copy backend files
COPY --chown=$UID:$GID ./backend .

EXPOSE 8080

HEALTHCHECK CMD curl --silent --fail http://localhost:${PORT:-8080}/health | jq -ne 'input.status == true' || exit 1

# Minimal, atomic permission hardening for OpenShift (arbitrary UID):
# - Group 0 owns /app and /root
# - Directories are group-writable and have SGID so new files inherit GID 0
RUN if [ "$USE_PERMISSION_HARDENING" = "true" ]; then \
    set -eux; \
    chgrp -R 0 /app /root || true; \
    chmod -R g+rwX /app /root || true; \
    find /app -type d -exec chmod g+s {} + || true; \
    find /root -type d -exec chmod g+s {} + || true; \
    fi

USER $UID:$GID

ARG BUILD_HASH
ENV WEBUI_BUILD_VERSION=${BUILD_HASH}
ENV DOCKER=true

CMD [ "bash", "start.sh"]
