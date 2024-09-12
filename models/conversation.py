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
    def get_model_name(cls) -> str:
        return cls.__modelname__

class ConversationSchema(PrimaryKeyMixinSchema, TimestampMixinSchema):
    uuid: str = Field(alias="sessionId", description='downcased alphanumeric session id')
    title: str = Field(description='title of conversation')
    messages: Optional[List[MessageSchema]] = Field(description='backward compatibility with chat-ui; not in use')
    message_ids: List[PyObjectId] = Field(description='bson object ids', default_factory=[])

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True

class CreateConversationSchema(ChatSchema):
    title: str = Field(description='title of conversation')
    message_ids: List[PyObjectId] = Field(description='bson object ids')

class ConversationIdSchema(ChatSchema):
    id: PyObjectId = Field(alias="_id", description='bson object id')

class UpdateConversationSchema(ChatSchema):
    title: Optional[str] = None
    message_ids: List[PyObjectId] = Field(description='bson object ids')
    updatedAt: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class ConversationCollectionSchema(ChatSchema):
    conversations: List[ConversationSchema]