from typing import List, TypedDict, Optional
from .abstract_model import AbstractModel
from .mongo_schema import Field

from .embedding_schema import EmbeddingBase

class EndpointDict(TypedDict):
    url: str
    type: str

class SystemEmbeddingConfig(AbstractModel):
    __modelname__ = 'system_embedding_configs'
    
    @classmethod
    def get_model_name(cls):
        return cls.__modelname__

class SystemEmbeddingConfigSchema(EmbeddingBase):
    endpoints: Optional[List[EndpointDict]] = Field(description='Valid if using TEI', default=None)
    
    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True