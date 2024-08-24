from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pymongo import ReturnDocument
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId
from chat_client import ChatClient as client
from models.bsonid import PyObjectId

class SettingModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", description='bson object id', default_factory=ObjectId)
    sessionId: str = Field(description='downcased alphanumeric session id')
    activeModel: str = Field(description='active model of user')
    customPrompts: Optional[Dict[str,str]] = Field(description='custom templates per model', default_factory=dict)
    hideEmojiOnSidebar: Optional[bool] = Field(description='hide emoji on sidebar', default=False)
    ethicsModalAcceptedAt: datetime = Field(default_factory=datetime.now)
    createdAt: datetime = Field(default_factory=datetime.now)
    updatedAt: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True

class SettingIdModel(BaseModel):
    id: PyObjectId = Field(alias="_id", description='bson object id')

class UpdateSettingModel(BaseModel):
    activeModel: Optional[str] = None
    customPrompts: Optional[Dict[str,str]] = None
    hideEmojiOnSidebar: Optional[bool] = None
    updatedAt: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class SettingCollection(BaseModel):
    messages: List[SettingModel]

class SettingFacade:
    @staticmethod
    def get_collection() -> AsyncIOMotorCollection:
        """Get the collection associated with Pydantic model"""
        return client.instance().db().get_collection('settings')
    
    @classmethod
    async def create(cls, uuid: str, setting: SettingModel):
        """"Create a new setting"""
        session_data = {**setting.model_dump(by_alias=True), 'sessionId': uuid}
        new_setting = await cls.get_collection().insert_one(session_data)
        return new_setting.inserted_id
    
    @classmethod
    async def find(cls, uuid: str, id: str) -> Dict[str, Any]:
        """"Find a setting by id"""
        return await cls.get_collection().find_one({"_id": ObjectId(id), 'sessionId': uuid})
    
    @classmethod
    async def update(cls, uuid: str, id: str, setting: UpdateSettingModel) -> Dict[str, Any]:
        """"Update a setting"""
        # keep only fields with values
        setting = {
            k: v for k, v in setting.model_dump(by_alias=True).items() if v is not None
        }
        if len(setting) >= 1:
            update_result = await cls.get_collection().find_one_and_update(
                {"_id": ObjectId(id), 'sessionId': uuid},
                {"$set": setting},
                return_document=ReturnDocument.AFTER,
            )
        return update_result
    
    @classmethod
    async def delete(cls, id: str) -> bool:
        """"Delete a setting"""
        delete_result = await cls.get_collection().delete_one({"_id": ObjectId(id)})
        if delete_result.deleted_count == 1:
            return True
        return False