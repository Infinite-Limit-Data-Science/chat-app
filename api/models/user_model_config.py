from datetime import datetime
from typing import List, Optional
from .abstract_model import AbstractModel
from .mongo_schema import (
    ChatSchema,
    Field,
    PyObjectId
)
from .llm_schema import (
    LLMBase,
    LLMParamSchema,
    PrimaryKeyMixinSchema, 
    TimestampMixinSchema,
)

class UserModelConfig(AbstractModel):
    __modelname__ = 'user_model_configs'
    
    @classmethod
    def get_model_name(cls):
        return cls.__modelname__

class UserModelConfigSchema(LLMBase, PrimaryKeyMixinSchema, TimestampMixinSchema):
    class Config:
        arbitrary_types_allowed = True

class UserModelConfigIdSchema(ChatSchema):
    id: PyObjectId = Field(alias="_id", description='bson object id')

class UpdateUserModelConfigSchema(LLMBase):
    parameters: Optional[LLMParamSchema] = Field(description='Parameters for the model', default=None)
    active: Optional[bool] = Field(description='Set model as active', default=None)
    updatedAt: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True

class UserModelConfigCollectionSchema(ChatSchema):
    user_model_configs: List[UserModelConfigSchema]