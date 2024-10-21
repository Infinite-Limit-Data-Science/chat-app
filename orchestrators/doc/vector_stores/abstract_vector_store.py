from typing import List, Any, Dict, TypedDict, Optional, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import reduce
from operator import and_
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from redisvl.query.filter import Tag, FilterExpression

class AbstractFlexiSchemaFields(TypedDict):
    name: str
    type: str
    value: str

class AbstractFlexiSchema(ABC):
    def __init__(self, schema: List[Dict[str, Any]]):
        self.schema = schema

    @abstractmethod
    def get_default_filter(self):
        pass

    @abstractmethod
    def get_all_filters(self):
        pass

@dataclass
class VectorStoreRetrieval:
    k: int = field(default=4)
    score_threshold: float = field(default=0.9)

class AbstractVectorStore(ABC):
    flexi_schema: AbstractFlexiSchema
    
    @abstractmethod
    async def aadd(self, documents: Iterator[Document]) -> List[str]:
        pass

    @abstractmethod
    async def asimilarity_search(
        self, 
        query: str, 
        *, 
        filter_expression: Optional[FilterExpression] = None) -> List[Document]:
        pass

    @abstractmethod
    def retriever(
        self, 
        options: VectorStoreRetrieval = VectorStoreRetrieval(),
        *,
        filter_expression: FilterExpression = None) -> VectorStoreRetriever:
        """Return a retriever Runnable to query vectorstore"""
        pass

    @abstractmethod
    async def inspect(
        self, 
        query: str,
        k: int = 4,
        *,
        filter_expression: Optional[FilterExpression] = None) -> str:
        """Inspect a query"""
        pass