from datetime import datetime
from typing import List, Dict, Union, Optional
from pydantic import BaseModel, Field
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
        return client.instance().db().get_collection('messages')
    
    @classmethod
    async def all(cls, conversation_id: str, limit: int, offset: int) -> List[MessageModel]:
        """Fetch all messages by conversation in database filtered by a limit and offset"""
        return await cls.get_collection().find({conversation_id}).skip(offset).limit(limit).to_list(limit)