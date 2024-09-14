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
    model_configs: List[PyObjectId] = Field(description='Model configs associated with the Prompt', default_factory=[])

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True

class PromptIdSchema(ChatSchema):
    id: PyObjectId = Field(alias="_id", description='bson object id')

class UpdatePromptSchema(ChatSchema):
    title: str = Field(description='Title of Prompt Template')
    prompt: str = Field(description='Content of Prompt Template')
    updatedAt: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class PromptCollectionSchema(ChatSchema):
    prompts: List[PromptSchema]