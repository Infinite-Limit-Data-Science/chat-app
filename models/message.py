from datetime import datetime
from typing import List, Dict, Union, Optional
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from objectid import PyObjectId
from chat_client import ChatClient as client

class Message(BaseModel):
	id: PyObjectId = Field(alias="_id")
	content: str
	modelDetail: Dict[str, Union[str, Dict[str, str]]]
	files: List[str]
	createdAt: datetime
	updatedAt: datetime