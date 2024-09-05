import os
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings as Embeddings
from langchain_redis import RedisConfig as Config, RedisVectorStore as VectorStore
from langchain_core.vectorstores import VectorStoreRetriever

SCHEMA = [
    {"name": "uuid", "type": "tag"},
    {"name": "conversation_id", "type": "tag"},
    {"name": "message_id", "type": "tag"}
]

embeddings = Embeddings(model_name=os.environ['EMBEDDINGS_MODEL'])

_config = Config(
    index_name='user_documents',
    redis_url = os.environ['REDIS_URL'],
    metadata_schema=SCHEMA,
)

class RediStore:
    class ConnectionException(Exception):
        def __init__(self, message='Connection not established'):
            super().__init__(message)

    _vector_store = None
    _exception = 'Connection not established'

    @classmethod
    def connect(cls) -> VectorStore:
        """Connect to vector store"""
        cls._vector_store = VectorStore(embeddings, config=_config)
        return cls._vector_store
    
    @classmethod
    def _disconnect(cls) -> None:
        """Disconnect from vector store (internal)"""
        cls._vector_store._index.client.quit()

    @classmethod
    def add(cls, documents: List[Document]) -> List[str]:
        """Add documents to the vector store, expecting metadata per document"""
        if not cls._vector_store:
            raise RediStore.ConnectionException()
        return cls._vector_store.add_documents(documents)
    
    @classmethod
    def similarity_search(cls, query: str, filter=filter) -> List[Document]:
        """Use Cosine Similarity Search to get immediate results"""
        if not cls._vector_store:
            raise RediStore.ConnectionException()
        results = cls._vector_store.similarity_search(query, filter=filter)
        return results
    
    @classmethod
    def as_retriever(cls, filter: dict={}) -> VectorStoreRetriever:
        """Interface for retrieving vectors from the vector store returning retriever object"""
        if not cls._vector_store:
            raise RediStore.ConnectionException()
        retriever = cls._vector_store.as_retriever(search_type="similarity", filter=filter)
        return retriever