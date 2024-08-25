import os
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings as Embeddings
from langchain_redis import RedisConfig as Config, RedisVectorStore as VectorStore

SCHEMA = [
    {"name": "uuid", "type": "tag"},
    {"name": "conversation_id", "type": "tag"},
    {"name": "message_id", "type": "tag"}
]

class RediStore:
    _instance = None

    @classmethod
    def instance(cls) -> 'RediStore':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.embeddings = Embeddings(model_name=os.environ['EMBEDDINGS_MODEL'])
        self.config = Config(
            index_name='user_documents',
            redis_url = os.environ['REDIS_URL'],
            metadata_schema=SCHEMA,
        )

    def connect(self) -> VectorStore:
        self.vector_store = VectorStore(self.embeddings, config=self.config)
    
    def _disconnect(self) -> None:
        self.vector_store._index.client.quit()

    def add(self, documents: List[Document]) -> List[str]:
        return self.vector_store.add_documents(documents)