from datetime import datetime
from typing import List, TypedDict
from models.abstract_model import AbstractModel
from models.mongo_schema import (
    ChatSchema,
    Field,
    PyObjectId
)
from models.llm_schema import LLMBase

class ModelConfig(AbstractModel):
    __modelname__ = 'model_configs'
    
    @classmethod
    def get_model_name(cls):
        return cls.__modelname__

class EndpointDict(TypedDict):
    url: str
    type: str

class ModelConfigEndpointSchema(LLMBase):
    endpoints: List[EndpointDict]
    
    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True

class ModelConfigPipelineSchema(LLMBase):
    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True

class ModelConfigIdSchema(ChatSchema):
    id: PyObjectId = Field(alias="_id", description='bson object id')

class UpdateModelConfigEndpointSchema(LLMBase):
    endpoints: List[EndpointDict]
    updatedAt: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class UpdateModelConfigPipelineSchema(LLMBase):
    updatedAt: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class ModelConfigEndpointCollectionSchema(ChatSchema):
    messages: List[ModelConfigEndpointSchema]

class ModelConfigPipelineSchemaCollectionSchema(ChatSchema):
    messages: List[ModelConfigPipelineSchema]