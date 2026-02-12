from open_webui.retrieval.vector.main import VectorDBBase
from open_webui.retrieval.vector.type import VectorType
from open_webui.config import (
    VECTOR_DB,
    ENABLE_QDRANT_MULTITENANCY_MODE,
    ENABLE_MILVUS_MULTITENANCY_MODE,
)
from open_webui.env import (
    CHROMA_REDIS_ENABLED,
    CHROMA_REDIS_URL,
    REDIS_CLUSTER,
    REDIS_KEY_PREFIX,
    REDIS_SENTINEL_HOSTS,
    REDIS_SENTINEL_PORT,
)
from open_webui.utils.redis import get_redis_connection, get_sentinels_from_env


class Vector:

    @staticmethod
    def get_vector(vector_type: str) -> VectorDBBase:
        """
        get vector db instance by vector type
        """
        match vector_type:
            case VectorType.MILVUS:
                if ENABLE_MILVUS_MULTITENANCY_MODE:
                    from open_webui.retrieval.vector.dbs.milvus_multitenancy import (
                        MilvusClient,
                    )

                    return MilvusClient()
                else:
                    from open_webui.retrieval.vector.dbs.milvus import MilvusClient

                    return MilvusClient()
            case VectorType.QDRANT:
                if ENABLE_QDRANT_MULTITENANCY_MODE:
                    from open_webui.retrieval.vector.dbs.qdrant_multitenancy import (
                        QdrantClient,
                    )

                    return QdrantClient()
                else:
                    from open_webui.retrieval.vector.dbs.qdrant import QdrantClient

                    return QdrantClient()
            case VectorType.PINECONE:
                from open_webui.retrieval.vector.dbs.pinecone import PineconeClient

                return PineconeClient()
            case VectorType.S3VECTOR:
                from open_webui.retrieval.vector.dbs.s3vector import S3VectorClient

                return S3VectorClient()
            case VectorType.OPENSEARCH:
                from open_webui.retrieval.vector.dbs.opensearch import OpenSearchClient

                return OpenSearchClient()
            case VectorType.PGVECTOR:
                from open_webui.retrieval.vector.dbs.pgvector import PgvectorClient

                return PgvectorClient()
            case VectorType.ELASTICSEARCH:
                from open_webui.retrieval.vector.dbs.elasticsearch import (
                    ElasticsearchClient,
                )

                return ElasticsearchClient()
            case VectorType.CHROMA:
                from open_webui.retrieval.vector.dbs.chroma import ChromaClient
                from open_webui.retrieval.vector.chroma_redis_coordinator import (
                    ChromaQueuedClient,
                    ChromaWriteCoordinator,
                )

                client = ChromaClient()

                if not CHROMA_REDIS_ENABLED:
                    return client

                if not CHROMA_REDIS_URL:
                    raise ValueError(
                        "CHROMA_REDIS_ENABLED=true requires CHROMA_REDIS_URL (or REDIS_URL)"
                    )

                redis_client = get_redis_connection(
                    redis_url=CHROMA_REDIS_URL,
                    redis_sentinels=get_sentinels_from_env(
                        REDIS_SENTINEL_HOSTS,
                        REDIS_SENTINEL_PORT,
                    ),
                    redis_cluster=REDIS_CLUSTER,
                    async_mode=False,
                )
                if redis_client is None:
                    raise ValueError(
                        "Failed to initialize Redis client for Chroma queued writes"
                    )

                coordinator = ChromaWriteCoordinator(
                    redis_client=redis_client,
                    key_prefix=REDIS_KEY_PREFIX,
                )
                return ChromaQueuedClient(inner_client=client, coordinator=coordinator)
            case VectorType.ORACLE23AI:
                from open_webui.retrieval.vector.dbs.oracle23ai import Oracle23aiClient

                return Oracle23aiClient()
            case VectorType.WEAVIATE:
                from open_webui.retrieval.vector.dbs.weaviate import WeaviateClient

                return WeaviateClient()
            case _:
                raise ValueError(f"Unsupported vector type: {vector_type}")


VECTOR_DB_CLIENT = Vector.get_vector(VECTOR_DB)
