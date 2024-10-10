import logging
import os
import asyncio
from itertools import islice
from typing import List, TypedDict, Any, Iterator, TypeVar, Dict, Literal, Union, Optional
from redis.connection import ConnectionPool
from langchain_core.documents import Document
from langchain_redis import RedisConfig as Config, RedisVectorStore as VectorStore
from langchain_redis.config import Redis
from langchain_core.vectorstores import VectorStoreRetriever
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from orchestrators.doc.vector_stores.abstract_vector_store import (
    AbstractVectorStore, 
    VectorStoreRetrieval,
    AbstractFlexiSchemaFields,
    AbstractFlexiSchema,
    FilterExpression,
    Tag,
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

_redis_client = Redis.from_pool(ConnectionPool.from_url(
    os.environ['REDIS_URL'], 
    max_connections=_MAX_CONNECTIONS,
    socket_timeout=_SOCKET_TIMEOUT))

T = TypeVar('T')

def chunked_iterable(it: Iterator[T], size: int) -> Iterator[list[T]]:
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

class RedisVectorStoreWithTTL(VectorStore):
    async def aadd_documents_with_ttl(
        self, 
        documents: list[Document], 
        ttl_seconds: int,
        batch_size: Literal[32, 64] = 64,
        **kwargs: Any) -> List[str]:
        """
        Example:
        > EXISTS user_conversations:a2c8a48073ee4a429b6910b1cfefb9f4
        (integer) 1
        > TTL user_conversations:a2c8a48073ee4a429b6910b1cfefb9f4
        (integer) 2591813
        """
        redis_client = self.config.redis()
        batches = chunked_iterable(documents, batch_size)
        tasks = [
            asyncio.create_task(self._process_batch(batch, ttl_seconds, redis_client, **kwargs))
            for batch in batches
        ]
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

class RedisConfig(TypedDict):
    index_name: str
    redis_url: str
    schema: List[dict]

class FlexiSchema(AbstractFlexiSchema):
    def __init__(self, schema: List[Dict[str, Any]]):
        self.schema = schema

    def _get_defaults(self) -> List[FilterExpression]:
        return [
            Tag(field['name']) == field['value']
            for field in self.schema
            if 'value' in field and field['value'] is not None
        ]

    def get_default_filter(self) -> FilterExpression:
        return self._combine_filters(self._get_defaults())

    def get_all_filters(
        self, 
        additional_filters: Dict[str, Any] = None) -> FilterExpression:
        filters = self._get_defaults()
        if additional_filters:
            for field in self.schema:
                field_name = field['name']
                if 'value' not in field and field_name in additional_filters:
                    filters.append(Tag(field_name) == additional_filters[field_name])

        return self._combine_filters(filters)

    @staticmethod
    def _combine_filters(filters: List[Any]) -> FilterExpression:
        if not filters:
            return None
        combined_filter = filters[0]
        for f in filters[1:]:
            combined_filter &= f
        return combined_filter

def _metadata_schema(schema):
    return [
        {key: field[key] for key in field if key != 'value'}
        for field in schema
    ]

class RedisVectorStore(AbstractVectorStore):
    def __init__(
        self, 
        embeddings: BaseEmbedding, 
        metadata: List[AbstractFlexiSchemaFields]):
        self._embeddings = embeddings
        self.flexi_schema = FlexiSchema(metadata)
        self._metadata_schema = _metadata_schema(metadata)
        self.config = Config(
            index_name=_INDEX_NAME,
            redis_client=_redis_client,
            metadata_schema=self._metadata_schema,
            distance_metric=_DISTANCE_METRIC,
            indexing_algorithm=_INDEXING_ALGORITHM,
            vector_datatype=_VECTOR_DATATYPE,
            storage_type=_STORAGE_TYPE,
            content_field=_CONTENT_FIELD_NAME,
            embedding_field=_EMBEDDING_VECTOR_FIELD_NAME,
            embedding_dimensions=self._embeddings.dimensions,
        )
        self._vector_store = RedisVectorStoreWithTTL(self._embeddings.endpoint_object, config=self.config)

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
    
    async def asimilarity_search(
        self, 
        query: str, 
        *, 
        filter_expression: Optional[FilterExpression] = None) -> List[Document]:
        """Use Async Cosine Similarity Search to get immediate results"""
        filter = filter_expression if filter_expression else self.flexi_schema.get_default_filter()
        logging.warning(f'WHAT ARE THE FILTERS APPLIED {filter}')
        return await self._vector_store.asimilarity_search(query, filter=filter)
    
    async def adelete(
        self, 
        query: str = '', 
        *, 
        filter_expression: Optional[FilterExpression] = None) -> bool:
        filter = filter_expression if filter_expression else self.flexi_schema.get_default_filter()
        documents = await self._vector_store.asimilarity_search(query=query, filter=filter)
        document_ids = [doc.metadata['id'] for doc in documents]
        if document_ids:
            result = self._vector_store.adelete(ids=document_ids)
            if result:
                logging.warning(f'Deleted documents on index {_INDEX_NAME}')
                return True
        return False
    
    def retriever(
        self, 
        options: VectorStoreRetrieval = VectorStoreRetrieval(),
        *,
        filter_expression: Optional[FilterExpression] = None) -> VectorStoreRetriever:
        """Generate a retriever from filter expression"""
        filter = filter_expression if filter_expression else self.flexi_schema.get_default_filter()
        retriever = self._vector_store.as_retriever(
            search_type='similarity',
            search_kwargs={
                'k': options.k, 
                'score_threshold': options.score_threshold, 
                'filter': filter,
            })
        return retriever

    async def inspect(
        self, 
        query: str, 
        k: int = 4, *, 
        filter_expression: Optional[FilterExpression] = None) -> str:
        from tabulate2 import tabulate
        query_vector = await self._embeddings.endpoint_object.aembed_query(query)
        filter = filter_expression if filter_expression else self.flexi_schema.get_default_filter()
        results = await self._vector_store.asimilarity_search_by_vector(
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
                *self._metadata_schema,
            ],
        })