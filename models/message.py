from datetime import datetime
from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel, Field
from pymongo import ReturnDocument
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId
from chat_client import ChatClient as client
from models.bsonid import PyObjectId

class MessageModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", description='bson object id', default_factory=ObjectId)
    content: str = Field(description='message content')
    modelDetail: Dict[str, Union[str, Dict[str, str]]]
    files: Optional[List[str]] = Field(description='file upload data', default=None)
    createdAt: datetime = Field(default_factory=datetime.now)
    updatedAt: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True

# class MessageIdModel(BaseModel):
#     id: PyObjectId = Field(alias="_id", description='bson object id')

class UpdateMessageModel(BaseModel):
    content: Optional[str] = None
    modelDetail: Optional[Dict[str, Union[str, Dict[str, str]]]] = None
    files: Optional[List[str]] = None
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
        message.id = ObjectId()
        update_result = await cls.get_collection().update_one(
            { "_id": ObjectId(conversation_id) },
            { "$push": { "messages": message.model_dump(by_alias=True) } }
        )
        if update_result.modified_count > 0:
            return message
        return None
    
    @classmethod
    async def find(cls, uuid: str, conversation_id: str, id: str) -> Dict[str, Any]:
        """"Find a message by id"""
        conversation = await cls.get_collection().find_one(
            {"_id": ObjectId(conversation_id), 'sessionId': uuid},
            { "messages": { "$elemMatch": { "_id": ObjectId(id) } } }
        )
        if conversation:
            return conversation['messages'][0]
        return None

    @classmethod
    async def update(cls, uuid: str, conversation_id: str, id: str, message: UpdateMessageModel) -> Optional[Dict[str,Any]]:
        """"Update a message"""
        message = {
            k: v for k, v in message.model_dump(by_alias=True).items() if v is not None
        }
        if len(message) >= 1:
            conversation = await cls.get_collection().find_one_and_update(
                {"_id": ObjectId(conversation_id), 'sessionId': uuid},
                {
                    "$set": {
                        f"messages.$[message].{k}": v
                        for k, v in message.items()
                    }
                },
                array_filters=[{"message._id": ObjectId(id)}],
                return_document=ReturnDocument.AFTER,
            )
            updated_message = next((msg for msg in conversation["messages"] if msg["_id"] == ObjectId(id)), None)
            return updated_message
        
    @classmethod
    async def delete(cls, uuid: str, conversation_id: str, id: str) -> bool:
        """"Delete a message"""
        delete_result = await cls.get_collection().update_one(
            {"_id": ObjectId(conversation_id), 'sessionId': uuid},
            {"$pull": {"messages": {"_id": ObjectId(id)}}}
        )
        if delete_result.modified_count == 1:
            return True
        return False