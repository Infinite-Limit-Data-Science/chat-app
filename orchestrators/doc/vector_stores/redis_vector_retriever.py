from typing import List, Self
from pydantic import Field, model_validator
from redisvl.query.filter import FilterExpression
from langchain_redis import RedisVectorStore
from orchestrators.doc.vector_stores.abstract_vector_retriever import AbstractVectorRetriever

class RedisVectorRetriever(AbstractVectorRetriever):
    filter: FilterExpression = Field(description='Filter expression for the retriever')
    tags: List[str] = Field(description='Tags to attach to Runnable', default=['redis', 'vectorstore', 'retriever'])

    @model_validator(mode='after')
    def load_retriever(self) -> Self:
        vector_store: RedisVectorStore = self.vector_store_proxy.vector_store
        self.runnable_name = '_'.join([v for _, v in self.metadata.items()])
        self.source = next((v for k, v in self.metadata.items() if k == 'source'), None)

        self.retriever = vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs={
                'k': self.k, 
                'score_threshold': self.score_threshold, 
                'filter': self.filter,                
            }).with_config(
                run_name=self.runnable_name,
                tags=self.tags,
                metadata=self.metadata,           
            )
            
        return self
    
    class Config:
        arbitrary_types_allowed = True