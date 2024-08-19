from datetime import datetime
from typing import List, Dict, Union, Optional
from pydantic import BaseModel, Field
from pymongo import ReturnDocument
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from bson import ObjectId
from chat_client import ChatClient as client
from models.bsonid import PyObjectId

class MessageModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", description='bson object id', default_factory=ObjectId)
    content: str = Field(description='the prompt sent to LLM')
    modelDetail: Dict[str, Union[str, Dict[str, str]]]
    files: Optional[List[str]] = Field(description='file upload data', default=None)
    createdAt: datetime = Field(default_factory=datetime.now)
    updatedAt: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True

class MessageIdModel(BaseModel):
    id: PyObjectId = Field(alias="_id", description='bson object id')

class UpdateMessageModel(BaseModel):
    content: Optional[str] = None
    updatedAt: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class MessageCollection(BaseModel):
    messages: List[MessageModel]

class MessageFacade:
    @staticmethod
    def get_collection() -> AsyncIOMotorCollection:
        """Get the collection associated with Pydantic model"""
        return client.instance().db().get_collection('conversations')
    
    @classmethod
    async def create(cls, conversation_id: str, message: MessageModel):
        """"Create a new message"""
        update_result = await cls.get_collection('conversations').updateOne(
            { "_id": ObjectId(conversation_id) },
            { "$push": { "messages": message } },
            return_document=ReturnDocument.AFTER
        )
        return update_result