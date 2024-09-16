from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from langchain_core.vectorstores import VectorStoreRetriever

@dataclass
class VectorStoreRetrieval:
    k: int = field(default=4)
    score_threshold: float = field(default=0.9)

class AbstractVectorStore(ABC):
    @abstractmethod
    def runnable(self, options: VectorStoreRetrieval = VectorStoreRetrieval()) -> VectorStoreRetriever:
        """Return a runnable of vector store object"""
        pass