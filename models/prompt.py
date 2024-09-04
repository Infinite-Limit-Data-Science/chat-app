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

class Prompt(AbstractModel):
    __modelname__ = 'prompts'
    
    @classmethod
    def get_model_name(cls):
        return cls.__modelname__
    
class PromptSchema(PrimaryKeyMixinSchema, TimestampMixinSchema):
    sessionId: str = Field(description='downcased alphanumeric session id')
    setting_id: Optional[PyObjectId] = Field(alias="_id", description='settings bson object id')
    title: str = Field(description='Title of Prompt Template')
    prompt: str = Field(description='Content of Prompt Template')

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True

class PromptIdSchema(ChatSchema):
    id: PyObjectId = Field(alias="_id", description='bson object id')

class UpdatePromptSchema(ChatSchema):
    setting_id: Optional[PyObjectId] = Field(alias="_id", description='settings bson object id')
    title: str = Field(description='Title of Prompt Template')
    prompt: str = Field(description='Content of Prompt Template')
    updatedAt: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class PromptCollectionSchema(ChatSchema):
    messages: List[PromptSchema]