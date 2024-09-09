from datetime import datetime
from typing import (
    List, 
    Dict, 
    Any, 
    Union, 
    Optional, 
    Literal, 
    TypedDict,
)
from models.abstract_model import AbstractModel
from models.mongo_schema import (
    ChatSchema,
    PrimaryKeyMixinSchema,
    TimestampMixinSchema,
    Field,
)

class Message(AbstractModel):
    __modelname__ = 'messages'

    @classmethod
    def get_model_name(cls):
        return cls.__modelname__

class AdditionalPayloadSchema(TypedDict):
    modelDetail: Dict[str, Union[str, Dict[str, str]]]
    files: Optional[List[str]] = Field(description='file upload data', default=None)

class BaseMessageSchema(TypedDict):
    content: str = Field(description='The corpus')
    type: Literal['SystemMessage', 'HumanMessage', 'AIMessage'] = Field(description='Origin of Message')
    additional_kwargs: AdditionalPayloadSchema = Field(description='Additional payload to store in message (backward compatibility)', default_factory={})

class MessageSchema(PrimaryKeyMixinSchema, TimestampMixinSchema):
    SessionId: str = Field(description='The session id of messages (RAG context), currently corresponding to conversation id')
    History: BaseMessageSchema = Field(description='shape of data for Conversational AI')

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True

class UpdateMessageSchema(ChatSchema):
    History: BaseMessageSchema
    updatedAt: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class MessageCollectionSchema(ChatSchema):
    messages: List[MessageSchema]