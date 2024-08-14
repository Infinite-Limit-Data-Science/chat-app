from datetime import datetime
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from objectid import PyObjectId

class User(BaseModel):
	id: PyObjectId = Field(alias="_id")
	sessionId: str
	createdAt: datetime
	updatedAt: datetime