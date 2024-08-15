from datetime import datetime
import re
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId
from pymongo import ReturnDocument
from chat_client import ChatClient as client
from models.bsonid import PyObjectId
from models.message import Message

class ConversationModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", description='bson object id', default=None)
    title: str = Field(description='title of conversation')
    rootMessageId: str = Field(description='the root message of the list of messages')
    messages: List[Message] = Field(description='list of messages associated with conversation')
    model: str = Field(description='LLM model name')
    preprompt: Optional[str] = Field(description='preprompt to send to LLM')
    createdAt: datetime
    updatedAt: datetime
    userAgent: Optional[str] = Field(description='browser user agent', default=None)
    embeddingModel: Optional[str] = Field(description='embedding model name', default=None)
    sessionId: str = Field(pre=lambda x: x.lower(), description='downcased alphanumeric session id (cannot be all numbers)')

    class Config:
        orm_mode = True
        populate_by_name = True
        arbitrary_types_allowed = True

    @field_validator('sessionId')
    def sessionId_validator(cls, value):
        """Validate sessionId is alphanumeric and not all numbers"""
        if not re.match('^(?=.*[a-zA-Z])[a-zA-Z0-9]+$', value):
            raise ValueError('Invalid pattern')
        return value
    
class UpdateConversationModel(BaseModel):
    title: Optional[str] = None
    messages: Optional[Message] = None
    updatedAt: datetime
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class ConversationCollection(BaseModel):
    conversations: List[ConversationModel]

class ConversationFacade:
    @staticmethod
    def get_collection() -> AsyncIOMotorCollection:
        """Get the collection associated with Pydantic model"""
        return client.instance().db().get_collection('conversations')

    @classmethod
    async def all(cls, limit: int, offset: int) -> List[ConversationModel]:
        """Fetch all conversations in database filtered by a limit and offset"""
        return await cls.get_collection().find().skip(offset).limit(limit).to_list(limit)

    @classmethod
    async def create(cls, conversation: ConversationModel):
        """"Create a new conversation"""
        new_conversation = await cls.get_collection().insert_one(
            conversation.model_dump(by_alias=True, exclude=["id"])
        )
        created_conversation = await cls.get_collection().find_one(
            {"_id": new_conversation.inserted_id}
        )
        return created_conversation
    
    @classmethod
    async def find(cls, id: int) -> ConversationModel:
        await cls.get_collection().find_one({"_id": ObjectId(id)})

    @classmethod
    async def update(cls, id: int, conversation: UpdateConversationModel) -> ConversationModel | None:
        conversation = {
            k: v for k, v in conversation.model_dump(by_alias=True).items() if v is not None
        }
        if len(conversation) >= 1:
            update_result = await cls.get_collection().find_one_and_update(
                {"_id": ObjectId(id)},
                {"$set": conversation},
                return_document=ReturnDocument.AFTER,
            )
        if update_result:
            return update_result
        return None