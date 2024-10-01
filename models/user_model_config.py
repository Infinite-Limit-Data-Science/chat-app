from datetime import datetime
from typing import List, TypedDict, Optional
from models.abstract_model import AbstractModel
from models.mongo_schema import (
    ChatSchema,
    Field,
    PyObjectId
)
from models.llm_schema import (
    LLMParamSchema,
    PrimaryKeyMixinSchema, 
    TimestampMixinSchema,
)

class UserModelConfig(AbstractModel):
    __modelname__ = 'user_model_configs'
    
    @classmethod
    def get_model_name(cls):
        return cls.__modelname__

class UserModelConfigSchema(PrimaryKeyMixinSchema, TimestampMixinSchema):
    name: str = Field(description='Name of model')
    active: bool = Field(description='Specify if the model is active', default=False)
    parameters: LLMParamSchema = Field(description='Parameters for the model')

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class UserModelConfigIdSchema(ChatSchema):
    id: PyObjectId = Field(alias="_id", description='bson object id')

class UpdateUserModelConfigSchema:
    parameters: LLMParamSchema = Field(description='Parameters for the model')
    updatedAt: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class UserModelConfigCollectionSchema(ChatSchema):
    user_model_configs: List[UserModelConfigSchema]