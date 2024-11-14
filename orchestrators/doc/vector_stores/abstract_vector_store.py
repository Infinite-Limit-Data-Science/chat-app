from typing import List, TypedDict, Iterator
from functools import reduce
import operator
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from redisvl.query.filter import Tag, FilterExpression

class SchemaField(TypedDict):
    name: str
    type: str

def create_filter_expression(
    vector_store_schema: List[SchemaField], 
    input_data: dict,
) -> FilterExpression:
    """Generate Filter Expression where values forced to str to potentially handle BSON object ids"""
    tag_expressions = [
        Tag(field['name']) == str(input_data[field['name']])
        for field in vector_store_schema
        if field['name'] in input_data and field['type'] == 'tag'
    ]
    
    filter_expression = reduce(operator.and_, tag_expressions)
    return filter_expression

class AbstractVectorStore(ABC):    
    @abstractmethod
    async def aadd(self, documents: Iterator[Document]) -> List[str]:
        pass

    @abstractmethod
    async def asimilarity_search(
        self, 
        query: str, 
        filter: FilterExpression = None
    ) -> List[Document]:
        pass

    @abstractmethod
    async def adelete(
        self, 
        query: str = '', 
        filter: FilterExpression = None
    ) -> bool:
        pass

    @abstractmethod
    async def inspect(
        self, 
        query: str,
        k: int = 4,
        filter: FilterExpression = None
    ) -> str:
        """Inspect a query"""
        pass