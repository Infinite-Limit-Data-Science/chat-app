from typing import List, TypedDict, Optional
from models.abstract_model import AbstractModel
from models.mongo_schema import Field

from models.embedding_schema import EmbeddingBase

class EndpointDict(TypedDict):
    url: str
    type: str

class EmbeddingConfig(AbstractModel):
    __modelname__ = 'embedding_configs'
    
    @classmethod
    def get_model_name(cls):
        return cls.__modelname__

class EmbeddingConfigSchema(EmbeddingBase):
    endpoints: Optional[List[EndpointDict]] = Field(description='Valid if using TEI', default=None)
    
    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True