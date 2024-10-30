import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
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
    title: Optional[str] = Field(description='title of conversation', default='')
    prompt_used: Optional[str] = Field(description='Prompt used for the given conversation', default='')
    model_name: Optional[str] = Field(description='Name of model used in conversation', default='')
    messages: List[MessageSchema] = Field(description='List of messages associated with conversation')
    filenames: Optional[List[str]] = Field(description='List of filenames used in conversation', default_factory=list)

class CreateConversationSchema(PrimaryKeyMixinSchema, TimestampMixinSchema):
    uuid: Optional[str] = Field(alias="sessionId", description='downcased alphanumeric session id', default=None)
    title: Optional[str] = Field(description='title of conversation', default=None)
    prompt_used: Optional[str] = Field(description='chat-ui legacy field', default='')
    model_name: Optional[str] = Field(description='Name of model used in conversation', default='')    
    message_ids: Optional[List[PyObjectId]] = Field(description='Messages associated with Conversation', default_factory=list)

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True

class ConversationCollectionSchema(ChatSchema):
    conversations: List[ConversationSchema]

class UpdateConversationSchema(ChatSchema):
    title: Optional[str] = None
    updatedAt: datetime = Field(default_factory=datetime.now)

class ConversationIdSchema(ChatSchema):
    id: PyObjectId = Field(alias="_id", description='bson object id')