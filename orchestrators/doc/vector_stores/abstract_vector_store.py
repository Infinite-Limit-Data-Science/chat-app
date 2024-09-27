from typing import List
from langchain_core.documents import Document
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from langchain_core.vectorstores import VectorStoreRetriever

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