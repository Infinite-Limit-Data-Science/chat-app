from datetime import datetime
from typing import List, Dict, Union, Optional
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from chat_client import ChatClient as client
from models.bsonid import PyObjectId

class Message(BaseModel):
	id: Optional[PyObjectId] = Field(alias="_id", description='bson object id', default=None)
	content: str
	modelDetail: Dict[str, Union[str, Dict[str, str]]]
	files: List[str]
	createdAt: datetime
	updatedAt: datetime