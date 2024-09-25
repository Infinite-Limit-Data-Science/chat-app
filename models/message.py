from typing import (
    List, 
    Optional,
)
from models.abstract_model import AbstractModel
from models.mongo_schema import (
    ChatSchema,
    PrimaryKeyMixinSchema,
    TimestampMixinSchema,
    Field,
    PyObjectId,
)

class Message(AbstractModel):
    __modelname__ = 'messages'

    @classmethod
    def get_model_name(cls):
        return cls.__modelname__

class MessageSchema(PrimaryKeyMixinSchema, TimestampMixinSchema):
    conversation_id: PyObjectId = Field(description='The session id of messages, a reference to conversation id')
    History: Optional[str] = Field(description='Shape of message history for Generative AI', default=None)
    type: str = Field(description='Type of message structure of Generative AI')
    content: str = Field(description='Message Content')

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True

class MessageCollectionSchema(ChatSchema):
    messages: List[MessageSchema]