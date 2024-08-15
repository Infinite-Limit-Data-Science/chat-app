from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from bsonid import PyObjectId

class Setting(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", description='bson object id', default=None)
    sessionId: str
    activeModel: str
    customPrompts: Optional[str]
    hideEmojiOnSidebar: bool
    ethicsModalAcceptedAt: datetime
    createdAt: datetime
    updatedAt: datetime

    class Config:
        orm_mode = True
        populate_by_name = True
        arbitrary_types_allowed = True
