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
    max_batch_tokens: int
    max_client_batch_size: int
    max_batch_requests: int
    num_workers: int   
    auto_truncate: bool
    token: str
    endpoint_object: AbstractEmbedding = field(init=False, repr=False)
