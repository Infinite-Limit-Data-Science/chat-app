from datetime import datetime
from typing import List, TypedDict, Optional
from models.abstract_model import AbstractModel
from models.mongo_schema import (
    ChatSchema,
    Field,
    PyObjectId
)
from models.llm_schema import LLMBase

class SystemModelConfig(AbstractModel):
    __modelname__ = 'system_model_configs'
    
    @classmethod
    def get_model_name(cls):
        return cls.__modelname__

class EndpointDict(TypedDict):
    url: str
    type: str

class SystemModelConfigSchema(LLMBase):
    endpoints: Optional[List[EndpointDict]] = Field(description='Valid if using TGI', default=None)
    model: Optional[str] = Field(description='Valid if using pipeline', default=None)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class SystemModelConfigIdSchema(ChatSchema):
    id: PyObjectId = Field(alias="_id", description='bson object id')

class UpdateSystemModelConfigSchema(LLMBase):
    endpoints: Optional[List[EndpointDict]]
    updatedAt: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class SystemModelConfigCollectionSchema(ChatSchema):
    system_model_configs: List[SystemModelConfigSchema]