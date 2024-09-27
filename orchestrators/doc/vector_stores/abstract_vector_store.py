from typing import List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import reduce
from operator import and_
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from redisvl.query.filter import Tag

@dataclass
class VectorStoreRetrieval:
    k: int = field(default=4)
    score_threshold: float = field(default=0.9)

class AbstractVectorStore(ABC):
    @abstractmethod
    async def aadd(self, documents: List[Document]) -> List[str]:
        pass

    @abstractmethod
    async def asimilarity_search(self, query: str, kwargs) -> List[Document]:
        pass

    @abstractmethod
    def retriever(self, options: VectorStoreRetrieval = VectorStoreRetrieval()) -> VectorStoreRetriever:
        """Return a runnable of vector store object (a retriever is a runnable)"""
        pass

    @abstractmethod
    async def inspect(self, query: str) -> str:
        """Inspect a query"""
        pass

    def generate_expression(self, wrapper_runnable_config):
        metadata = wrapper_runnable_config['metadata']
        schema_keys = [item['name'] for item in metadata['schema']]
        filter_expression = reduce(
            and_,
            [
                (Tag(key) == str(metadata[key]))
                for key in schema_keys
            ]
        )
        return filter_expression