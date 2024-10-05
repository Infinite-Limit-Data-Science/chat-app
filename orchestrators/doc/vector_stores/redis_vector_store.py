import logging
import os
import asyncio
from typing import List, TypedDict, Any
from redis.connection import ConnectionPool
from langchain_core.documents import Document
from langchain_redis import RedisConfig as Config, RedisVectorStore as VectorStore
from langchain_redis.config import Redis
from langchain_core.vectorstores import VectorStoreRetriever
from orchestrators.doc.vector_stores.abstract_vector_store import (
    AbstractVectorStore, 
    VectorStoreRetrieval,
)
from orchestrators.doc.embedding_models.model_proxy import ModelProxy
from orchestrators.doc.runnable_extensions.wrapper_runnable_config import WrapperRunnableConfig

_MAX_CONNECTIONS = 50

_SOCKET_TIMEOUT = 30.0

_VECTOR_TTL_30_DAYS = 3600 * 24 * 30

_INDEX_NAME = 'user_conversations'

_DISTANCE_METRIC = 'COSINE'

_INDEXING_ALGORITHM = 'FLAT'

_VECTOR_DATATYPE = 'FLOAT32'

_STORAGE_TYPE = 'hash'

_CONTENT_FIELD_NAME = 'text'

_EMBEDDING_VECTOR_FIELD_NAME = 'embedding'

_redis_client = Redis.from_pool(ConnectionPool.from_url(
    os.environ['REDIS_URL'], 
    max_connections=_MAX_CONNECTIONS,
    socket_timeout=_SOCKET_TIMEOUT))

class RedisVectorStoreWithTTL(VectorStore):
    async def aadd_documents_with_ttl(
            self, 
            documents: list[Document], 
            ttl_seconds: int, 
            **kwargs: Any) -> List[str]:
        """
        Example:
        > EXISTS user_conversations:a2c8a48073ee4a429b6910b1cfefb9f4
        (integer) 1
        > TTL user_conversations:a2c8a48073ee4a429b6910b1cfefb9f4
        (integer) 2591813
        """
        document_ids = await self.aadd_documents(documents, **kwargs)
        redis_client = self.config.redis()
        for doc_id in document_ids:
            await asyncio.to_thread(redis_client.expire, doc_id, ttl_seconds)
        return document_ids

class RedisConfig(TypedDict):
    index_name: str
    redis_url: str
    schema: List[dict]

class RedisVectorStore(AbstractVectorStore):
    def __init__(
            self, 
            embeddings: ModelProxy, 
            wrapper_runnable_config: WrapperRunnableConfig):
        self._primary_embedding = embeddings.get()
        self._wrapper_runnable_config = wrapper_runnable_config
        self.config = Config(
            index_name=_INDEX_NAME,
            redis_client=_redis_client,
            metadata_schema=self._wrapper_runnable_config['metadata']['schema'],
            distance_metric=_DISTANCE_METRIC,
            indexing_algorithm=_INDEXING_ALGORITHM,
            vector_datatype=_VECTOR_DATATYPE,
            storage_type=_STORAGE_TYPE,
            content_field=_CONTENT_FIELD_NAME,
            embedding_field=_EMBEDDING_VECTOR_FIELD_NAME,
            embedding_dimensions=self._primary_embedding.dimensions,
        )
        self._vector_store = RedisVectorStoreWithTTL(self._primary_embedding.endpoint_object, config=self.config)

    @property
    def content_field_name(self) -> str:
        """Name for document content"""
        return self.config.content_field
    
    @property
    def embedding_vector_field_name(self) -> str:
        """Name for embedding vectors"""
        return self.config.embedding_field
    
    @property
    def embedding_dimensions(self) -> int:
        """Embedding Dimension count"""
        return self.config.embedding_dimensions

    async def aadd(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store asynchronously, expecting metadata per document"""
        return await self._vector_store.aadd_documents_with_ttl(documents, _VECTOR_TTL_30_DAYS)
    
    async def asimilarity_search(self, query: str) -> List[Document]:
        """Use Async Cosine Similarity Search to get immediate results"""
        filter_expression = self.generate_expression(self._wrapper_runnable_config)
        return await self._vector_store.asimilarity_search(query, filter=filter_expression)
    
    async def adelete(self, query: str = '') -> bool:
        filter_expression = self.generate_expression(self._wrapper_runnable_config)
        documents = await self._vector_store.asimilarity_search(query=query, filter=filter_expression)
        document_ids = [doc.metadata['id'] for doc in documents]
        if document_ids:
            result = self._vector_store.adelete(ids=document_ids)
            if result:
                logging.warning(f'Deleted documents on index {_INDEX_NAME}')
                return True
        return False
    
    def retriever(self, options: VectorStoreRetrieval = VectorStoreRetrieval()) -> VectorStoreRetriever:
        """Generate a retriever which implements the Runnable interface"""
        filter_expression = self.generate_expression(self._wrapper_runnable_config)
        retriever = self._vector_store.as_retriever(
            search_type='similarity',
            search_kwargs={
                'k': options.k, 
                'score_threshold': options.score_threshold, 
                'filter': filter_expression,
            })
        return retriever

    async def inspect(self, query: str, k: int = 4) -> str:
        from tabulate import tabulate
        query_vector = await self._primary_embedding.endpoint_object.aembed_query(query)
        filter_expression = self.generate_expression(self._wrapper_runnable_config)
        results = await self._vector_store.asimilarity_search_by_vector(
            embedding=query_vector,
            k=k,
            filter=filter_expression,
        )
        table_data = []
        for result in results:
            table_data.append([result.page_content, result.metadata])
        headers = ['Document', 'Metadata']

        output = f"""
            Document Size:
            {len(results)}
            Data: 
            {tabulate(table_data, headers=headers, tablefmt="grid")}
            Schema: 
            {self}
        """

        logging.warning(output)
        return output

    def __str__(self):
        """Index Schema"""
        return str({
            'index': {
                'name': self.config.index_name,
                'prefix': self.config.key_prefix,
                'storage_type': self.config.storage_type,
            },
            'fields': [
                {
                    'name': self.content_field_name,
                    'type': 'text'
                },
                {
                    'name': self.embedding_vector_field_name,
                    'type': 'vector',
                    'attrs': {
                        'dims': self.config.embedding_dimensions,
                        'distance_metric': self.config.distance_metric,
                        'algorithm': self.config.indexing_algorithm,
                        'datatype': self.config.vector_datatype,
                    },
                },
                *self._wrapper_runnable_config['metadata']['schema'],
            ],
        })