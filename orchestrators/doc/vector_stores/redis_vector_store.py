import os
from functools import reduce
from operator import and_
from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_redis import RedisConfig as Config, RedisVectorStore as VectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from redisvl.query.filter import Tag
from orchestrators.doc.vector_stores.abstract_vector_store import (
    AbstractVectorStore, 
    VectorStoreRetrieval,
)
from orchestrators.doc.embedding_models.model_proxy import ModelProxy
from orchestrators.doc.runnable_extensions.wrapper_runnable_config import WrapperRunnableConfig

_INDEX_NAME = 'user_conversations'

class RedisConfig(TypedDict):
    index_name: str
    redis_url: str
    schema: List[dict]

class RedisVectorStore(AbstractVectorStore):
    def __init__(
            self, 
            embeddings: ModelProxy, 
            wrapper_runnable_config: WrapperRunnableConfig):
        self._embeddings = embeddings
        self._wrapper_runnable_config = wrapper_runnable_config
        config = Config(
            index_name =_INDEX_NAME,
            redis_url = os.environ['REDIS_URL'],
            metadata_schema=self._wrapper_runnable_config['metadata']['schema']
        )
        self._vector_store = VectorStore(self._embeddings.get().endpoint_object, config=config)

    async def aadd(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store asynchronously, expecting metadata per document"""
        return await self._vector_store.aadd_documents(documents)
    
    async def asimilarity_search(self, query: str) -> List[Document]:
        """Use Async Cosine Similarity Search to get immediate results"""
        metadata = self._wrapper_runnable_config['metadata']
        schema_keys = [item['name'] for item in metadata['schema']]
        filter_expression = reduce(
            and_,
            [
                (Tag(key) == str(metadata[key]))
                for key in schema_keys
            ]
        )
        results = await self._vector_store.asimilarity_search(query, filter=filter_expression)
        return results
    
    def retriever(self, options: VectorStoreRetrieval = VectorStoreRetrieval()) -> VectorStoreRetriever:
        """Generate a retriever which implements the Runnable interface"""
        vector_filter = {
            item['name']: str(self._wrapper_runnable_config['metadata'][item['name']])
            for item in self._wrapper_runnable_config['metadata']['schema']
        }
        retriever = self._vector_store.as_retriever(
            search_type='similarity', 
            k=options.k, 
            score_threshold=options.score_threshold, 
            filter=vector_filter)
        return retriever