from datetime import datetime
from typing import List, Optional
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
    def get_collection_name(cls):
        return cls.__modelname__

class ConversationSchema(ChatSchema, PrimaryKeyMixinSchema, TimestampMixinSchema):
    title: str = Field(description='title of conversation')
    # rootMessageId: str = Field(description='the root message of the list of messages')
    messages: List[MessageSchema] = Field(description='list of messages associated with conversation')
    model: str = Field(description='LLM model name')
    preprompt: Optional[str] = Field(description='preprompt to send to LLM', default=None)
    userAgent: Optional[str] = Field(description='browser user agent', default=None)
    embeddingModel: Optional[str] = Field(description='embedding model name', default=None)
    uuid: str = Field(alias="sessionId", description='downcased alphanumeric session id')

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True

class CreateConversationSchema(ChatSchema):
    title: str = Field(description='title of conversation')
    messages: List[MessageSchema] = Field(description='list of messages associated with conversation')
    model: str = Field(description='LLM model name')
    preprompt: Optional[str] = Field(description='preprompt to send to LLM', default=None)

class ConversationIdSchema(ChatSchema):
    id: PyObjectId = Field(alias="_id", description='bson object id')

class UpdateConversationSchema(ChatSchema):
    title: Optional[str] = None
    messages: Optional[MessageSchema] = None
    updatedAt: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class ConversationCollectionSchema(ChatSchema):
    conversations: List[ConversationSchema]