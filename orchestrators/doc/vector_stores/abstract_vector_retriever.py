from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.vectorstores import VectorStoreRetriever
from orchestrators.doc.vector_stores.abstract_vector_store import AbstractVectorStore

class AbstractVectorRetriever(BaseModel):
    vector_store_proxy: AbstractVectorStore = Field(description='Vector Store Proxy')
    retriever: Optional[VectorStoreRetriever] = Field(description='Vector Store Retriever', default=None)
    metadata: Optional[dict] = Field(description='Metadata to attach to Runnable', default={})
    k: Optional[int] = Field(description='k number of results', default=4)
    score_threshold: Optional[float] = Field(description='score threshold', default=0.9)
    search_type: Optional[str] = Field(description='Vector search, one of cosine similarity or Euclidean distance', default='similarity')
    source: Optional[str] = Field(description='Source associated with the vectorized content', default=None)
    runnable_name: Optional[str] = Field(description='Name of the retriever object that implements Runnable Interface', default=None)

    class Config:
        arbitrary_types_allowed = True