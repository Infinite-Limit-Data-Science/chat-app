import logging
import os
import asyncio
from typing import List, Any, Iterator, Dict, Optional
from redis.client import Redis
from redis.connection import ConnectionPool

from langchain_core.documents import Document
from langchain_redis import RedisConfig as Config
from langchain_redis.config import Redis
from langchain_redis import RedisVectorStore

from ..embedding_models.embedding import BaseEmbedding
from .abstract_vector_store import (
    AbstractVectorStore, 
    FilterExpression,
)

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

class RedisVectorProxy(AbstractVectorStore):
    """
    Proxy to RedisVectorStore
    
    Should only be instantiated once for lifecycle of application
    """
    class MyRedisVectorStore(RedisVectorStore):
        async def aadd_documents_with_ttl(
            self, 
            documents: Iterator[Document], 
            ttl_seconds: int,
            max_requests: int,
            **kwargs: Any) -> List[str]:
            """
            Example:
            > EXISTS user_conversations:a2c8a48073ee4a429b6910b1cfefb9f4
            (integer) 1
            > TTL user_conversations:a2c8a48073ee4a429b6910b1cfefb9f4
            (integer) 2591813
            """
            redis_client = self.config.redis()

            semaphore = asyncio.Semaphore(max_requests)

            async def process_document(document: Document):
                async with semaphore:
                    batch_ids = await self._process_batch([document], ttl_seconds, redis_client, **kwargs)
                    return batch_ids
            
            tasks = [asyncio.create_task(process_document(document)) for document in documents]
            results = await asyncio.gather(*tasks)
            document_ids = [doc_id for batch_ids in results for doc_id in batch_ids]

            return document_ids
        
        async def _process_batch(
            self, 
            batch: List[Document], 
            ttl_seconds: int, 
            redis_client, 
            **kwargs: Any) -> List[str]:
            batch_ids = await self.aadd_documents(batch, **kwargs)
            for doc_id in batch_ids:
                await asyncio.to_thread(redis_client.expire, doc_id, ttl_seconds)

            return batch_ids

    def __init__(self, client: Redis, embeddings: BaseEmbedding, schema: List[Dict[str, Any]]):
        self._client = client
        self.embeddings = embeddings
        self._schema = schema
        self.config = Config(
            index_name=_INDEX_NAME,
            redis_client=self._client,
            metadata_schema=self._schema,
            distance_metric=_DISTANCE_METRIC,
            indexing_algorithm=_INDEXING_ALGORITHM,
            vector_datatype=_VECTOR_DATATYPE,
            storage_type=_STORAGE_TYPE,
            content_field=_CONTENT_FIELD_NAME,
            embedding_field=_EMBEDDING_VECTOR_FIELD_NAME,
            embedding_dimensions=self.embeddings.dimensions,
        )
        self.vector_store = RedisVectorProxy.MyRedisVectorStore(
            self.embeddings.endpoint_object, config=self.config)

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

    def update_embedding_token(self, new_token: str) -> None:
        self.embeddings.update_token(new_token)
        self.vector_store = RedisVectorProxy.MyRedisVectorStore(
                self.embeddings.endpoint_object, config=self.config)

    async def aadd(self, documents: Iterator[Document]) -> List[str]:
        """Add documents to the vector store asynchronously, expecting metadata per document"""
        return await self.vector_store.aadd_documents_with_ttl(documents, _VECTOR_TTL_30_DAYS, self.embeddings.max_batch_requests)
    
    async def asimilarity_search(
        self, 
        query: str,
        filter: FilterExpression = None
    ) -> List[Document]:
        """Use Async Cosine Similarity Search to get immediate results"""
        return await self.vector_store.asimilarity_search(query, filter=filter)
    
    async def adelete(
        self, 
        query: str = '',
        filter: FilterExpression = None
    ) -> bool:
        documents = await self.vector_store.asimilarity_search(query=query, filter=filter)
        document_ids = [doc.metadata['id'] for doc in documents]
        if document_ids:
            result = self.vector_store.adelete(ids=document_ids)
            if result:
                logging.warning(f'Deleted documents on index {_INDEX_NAME}')
                return True
        return False
    
    async def inspect(
        self, 
        query: str, 
        k: int = 4,
        filter: FilterExpression = None
    ) -> str:
        from tabulate2 import tabulate
        query_vector = await self.embeddings.endpoint_object.aembed_query(query)
        results = await self.vector_store.asimilarity_search_by_vector(
            embedding=query_vector,
            k=k,
            filter=filter,
        )
        table_data = []
        for result in results:
            table_data.append([result.page_content, result.metadata])
        headers = ['Document', 'Metadata']

        output = f"""
            Document Size:
            {len(results)}
            Data: 
            {tabulate(table_data, headers=headers, tablefmt='grid')}
            Schema: 
            {self}
        """

        logging.info(output)
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
                *self._schema,
            ],
        })

if not os.environ['REDIS_URL']:
    raise Exception('Missing `REDIS_URL` in environment, therefore, not trying to connect')

_redis_client = Redis.from_pool(ConnectionPool.from_url(
    os.environ['REDIS_URL'], 
    max_connections=_MAX_CONNECTIONS,
    socket_timeout=_SOCKET_TIMEOUT))

_redis_vector_instance: Optional[RedisVectorProxy] = None

def create_redis_vector_proxy(
    vector_store_schema: List[Dict[str, Any]],
    embeddings: BaseEmbedding,
) -> RedisVectorProxy:
    global _redis_vector_instance

    if _redis_vector_instance is None:
        _redis_vector_instance = RedisVectorProxy(
            client=_redis_client,
            embeddings=embeddings,
            schema=vector_store_schema,
        )
    else:
        _redis_vector_instance.update_embedding_token(embeddings.token)
    
    return _redis_vector_instance