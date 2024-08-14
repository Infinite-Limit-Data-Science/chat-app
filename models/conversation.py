from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from objectid import PyObjectId
from chat_client import ChatClient as client
from message import Message

class Conversation(BaseModel):
	id: PyObjectId = Field(alias="_id")
	title: str
	rootMessageId: str
	messages: List[Message]
	model: str
	preprompt: Optional[str]
	createdAt: datetime
	updatedAt: datetime
	userAgent: str
	embeddingModel: str
	sessionId: str

	@staticmethod
	def get_collection() -> AsyncIOMotorCollection:
		db: AsyncIOMotorDatabase = client.db()
		return db.get_collection('conversations')

	async def all(cls, limit: int, offset: int) -> List[Conversation]:
		conversations = []
		cursor = await cls.get_collection().find({ limit: limit, offset: offset })
		async for conversation in cursor:
			conversations.append(conversation)
		return conversations