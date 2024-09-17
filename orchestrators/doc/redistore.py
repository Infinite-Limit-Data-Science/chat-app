import os
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings as Embeddings
from langchain_redis import RedisConfig as Config, RedisVectorStore as VectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from redisvl.query.filter import Tag
from orchestrators.doc.abstract_vector_store import AbstractVectorStore, VectorStoreRetrieval

SCHEMA = [
    {"name": "uuid", "type": "tag"},
    {"name": "conversation_id", "type": "tag"},
]

embeddings = Embeddings(model_name=os.environ['EMBEDDINGS_MODEL'])

_config = Config(
    index_name='user_documents',
    redis_url = os.environ['REDIS_URL'],
    metadata_schema=SCHEMA,
)

class RediStore(AbstractVectorStore):
    class ConnectionException(Exception):
        def __init__(self, message='Connection not established'):
            super().__init__(message)

    _vector_store = None
    _exception = 'Connection not established'

    def __init__(self, uuid: str, conversation_id: str):
        self.uuid = uuid
        self.conversation_id = conversation_id

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
    async def add(cls, documents: List[Document]) -> List[str]:
        """Add documents to the vector store, expecting metadata per document"""
        if not cls._vector_store:
            raise RediStore.ConnectionException()
        return await cls._vector_store.aadd_documents(documents)
    
    @classmethod
    def similarity_search(cls, query: str, kwargs) -> List[Document]:
        """Use Cosine Similarity Search to get immediate results"""
        """It's recommended to use runnable instead"""
        if not cls._vector_store:
            raise RediStore.ConnectionException()
        
        filter = (Tag("uuid") == kwargs['uuid']) & (Tag("conversation_id") == str(kwargs['conversation_id']))
        results = cls._vector_store.similarity_search(query, filter=filter)
        return results
    
    def runnable(self, options: VectorStoreRetrieval = VectorStoreRetrieval()) -> VectorStoreRetriever:
        """Generate a retriever which is a runnable to be incorporated in chain"""
        vector_filter = { 
            'uuid': self.uuid, 
            'conversation_id': self.conversation_id,
        }
        retriever = self._vector_store.as_retriever(search_type="similarity", k=options.k, score_threshold=options.score_threshold, filter=vector_filter)
        return retriever