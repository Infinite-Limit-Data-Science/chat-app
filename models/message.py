from datetime import datetime
from typing import List, Dict, Any, Union, Optional
from pymongo import ReturnDocument
from models.abstract_model import AbstractModel
from models.message import MessageSchema
from models.mongo_schema import (
    ChatSchema,
    PrimaryKeyMixinSchema,
    TimestampMixinSchema,
    Field,
)

class Message(AbstractModel):
    __modelname__ = 'messages'

    @classmethod
    def get_collection_name(cls):
        return cls.__modelname__

class MessageSchema(ChatSchema, PrimaryKeyMixinSchema, TimestampMixinSchema):
    content: str = Field(description='message content')
    modelDetail: Dict[str, Union[str, Dict[str, str]]]
    files: Optional[List[str]] = Field(description='file upload data', default=None)

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True

# class MessageIdSchema(ChatSchema):
#     id: PyObjectId = Field(alias="_id", description='bson object id')

class UpdateMessageSchema(ChatSchema):
    content: Optional[str] = None
    modelDetail: Optional[Dict[str, Union[str, Dict[str, str]]]] = None
    files: Optional[List[str]] = None
    updatedAt: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class MessageCollectionSchema(ChatSchema):
    messages: List[MessageSchema]