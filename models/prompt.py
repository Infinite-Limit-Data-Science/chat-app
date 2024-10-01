from datetime import datetime
from typing import List, Dict, Optional
from models.abstract_model import AbstractModel
from models.mongo_schema import (
    ChatSchema,
    PrimaryKeyMixinSchema,
    TimestampMixinSchema,
    Field,
    PyObjectId
)
from models.llm_schema import LLMPrompt
    
class Prompt(AbstractModel):
    __modelname__ = 'prompts'
    
    @classmethod
    def get_model_name(cls):
        return cls.__modelname__
    
class PromptSchema(LLMPrompt, PrimaryKeyMixinSchema, TimestampMixinSchema):
    user_model_configs: List[PyObjectId] = Field(description='Associate zero, one, or many user model configs with a prompt', default_factory=list)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class PromptIdSchema(ChatSchema):
    id: PyObjectId = Field(alias="_id", description='bson object id')

class UpdatePromptSchema(LLMPrompt):
    updatedAt: datetime = Field(default_factory=datetime.now)

class PromptCollectionSchema(ChatSchema):
    prompts: List[PromptSchema]