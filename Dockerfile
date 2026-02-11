# syntax=docker/dockerfile:1
# Initialize device type args
# use build args in the docker build command with --build-arg="BUILDARG=true"
ARG USE_CUDA=false
ARG USE_SLIM=false
ARG USE_PERMISSION_HARDENING=false
# Tested with cu117 for CUDA 11 and cu121 for CUDA 12 (default)
ARG USE_CUDA_VER=cu128
ARG USE_EMBEDDING_MODEL=""
ARG USE_RERANKING_MODEL=""

# Tiktoken encoding name; models to use can be found at https://huggingface.co/models?library=tiktoken
# ARG USE_TIKTOKEN_ENCODING_NAME="cl100k_base"
ARG USE_TIKTOKEN_ENCODING_NAME=""

ARG BUILD_HASH=dev-build
ARG INSTALL_OPTIONAL_DEPS=true
ARG INSTALL_DEV_DEPS=false
# Override at your own risk - non-root configurations are untested
ARG UID=0
ARG GID=0

######## WebUI frontend ########
FROM --platform=$BUILDPLATFORM node:22-alpine3.20 AS frontend-build
ARG BUILD_HASH

# Set Node.js options (heap limit Allocation failed - JavaScript heap out of memory)
# ENV NODE_OPTIONS="--max-old-space-size=4096"

WORKDIR /app

# to store git revision in build
RUN apk add --no-cache git

# Improve npm install resilience in CI/builders and skip ORT CUDA binary download.
ENV npm_config_onnxruntime_node_install_cuda=skip \
    npm_config_fetch_retries=5 \
    npm_config_fetch_retry_mintimeout=20000 \
    npm_config_fetch_retry_maxtimeout=120000 \
    npm_config_network_timeout=600000

COPY package.json package-lock.json .npmrc ./
RUN npm ci --force

# Copy only frontend build inputs to reduce context churn
COPY src ./src
COPY static ./static
COPY scripts ./scripts
COPY svelte.config.js vite.config.ts tsconfig.json postcss.config.js tailwind.config.js ./

ENV APP_BUILD_HASH=${BUILD_HASH}
RUN npm run build

######## Python deps builder ########
FROM python:3.11-slim-bookworm AS backend-builder
ARG INSTALL_OPTIONAL_DEPS
ARG INSTALL_DEV_DEPS

WORKDIR /app/backend

ENV PYTHONUNBUFFERED=1

# Install build-only dependencies for Python wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements-min.txt backend/requirements-extra.txt backend/requirements-dev.txt ./

RUN pip3 install --no-cache-dir uv && \
    uv pip install --system -r requirements-min.txt --no-cache-dir && \
    if [ "$INSTALL_OPTIONAL_DEPS" = "true" ]; then \
      uv pip install --system -r requirements-extra.txt --no-cache-dir; \
    fi && \
    if [ "$INSTALL_DEV_DEPS" = "true" ]; then \
      uv pip install --system -r requirements-dev.txt --no-cache-dir; \
    fi

FROM backend-builder AS backend-builder-dev
RUN uv pip install --system -r requirements-dev.txt --no-cache-dir

######## WebUI runtime ########
FROM python:3.11-slim-bookworm AS runtime

# Use args
# ARG USE_CUDA
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

## Basis URL Config ##
# ENV OPENAI_API_BASE_URL=""

## API Key and Security Config ##
ENV OPENAI_API_KEY="" \
    WEBUI_SECRET_KEY="" \
    SCARF_NO_ANALYTICS=true \
    DO_NOT_TRACK=true \
    ANONYMIZED_TELEMETRY=false

WORKDIR /app/backend

ENV HOME=/root
RUN if [ $UID -ne 0 ]; then \
      if [ $GID -ne 0 ]; then \
        addgroup --gid $GID app; \
      fi; \
      adduser --uid $UID --gid $GID --home $HOME --disabled-password --no-create-home app; \
    fi && \
    mkdir -p $HOME/.cache/chroma && \
    echo -n 00000000-0000-0000-0000-000000000000 > $HOME/.cache/chroma/telemetry_user_id && \
    chown -R $UID:$GID /app $HOME

# Install runtime system dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git pandoc curl jq \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy prebuilt Python environment from dependency builder
COPY --from=backend-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# copy built frontend files
COPY --chown=$UID:$GID --from=frontend-build /app/build /app/build
COPY --chown=$UID:$GID ./CHANGELOG.md /app/CHANGELOG.md
COPY --chown=$UID:$GID ./package.json /app/package.json

# copy backend files
COPY --chown=$UID:$GID ./backend .

EXPOSE 8080

HEALTHCHECK CMD curl --silent --fail http://localhost:${PORT:-8080}/health | jq -ne "input.status == true" || exit 1

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

CMD ["bash", "start.sh"]

FROM runtime AS test
COPY --from=backend-builder-dev /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-builder-dev /usr/local/bin /usr/local/bin

FROM runtime AS production
