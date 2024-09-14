import logging
from datetime import datetime
from typing import List, Optional, Any
from bson import ObjectId
from pydantic import field_validator, model_validator
from models.abstract_model import AbstractModel
from models.message import MessageSchema
from models.mongo_schema import (
    ChatSchema,
    PrimaryKeyMixinSchema,
    TimestampMixinSchema,
    Field,
    PyObjectId
)

class Conversation(AbstractModel):
    __modelname__ = 'conversations'

    @classmethod
    def get_model_name(cls) -> str:
        return cls.__modelname__

class ConversationSchema(PrimaryKeyMixinSchema, TimestampMixinSchema):
    uuid: Optional[str] = Field(alias="sessionId", description='downcased alphanumeric session id', default=None)
    title: str = Field(description='title of conversation')
    message_ids: Optional[List[PyObjectId]] = Field(description='Messages associated with Conversation', default=None)

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True

class ConversationIdSchema(ChatSchema):
    id: PyObjectId = Field(alias="_id", description='bson object id')

class UpdateConversationSchema(ChatSchema):
    title: Optional[str] = None
    updatedAt: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class ConversationCollectionSchema(ChatSchema):
    conversations: List[ConversationSchema]