from typing import TypedDict
from dataclasses import dataclass, field
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

class AbstractEmbedding(BaseModel, Embeddings):
    pass

class EndpointDict(TypedDict):
    url: str
    type: str

@dataclass
class BaseEmbedding:
    name: str
    description: str
    task: str
    endpoint: EndpointDict
    dimensions: int
    token: str
    endpoint_object: AbstractEmbedding = field(init=False, repr=False)
