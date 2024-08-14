import os
from motor import motor_asyncio
from motor.motor_asyncio import AsyncIOMotorDatabase

class ChatClient:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.client = None

    async def connect(self, url):
        self.client = motor_asyncio.AsyncIOMotorClient(url)

    async def close(self):
        self.client.close()
    
    def db(self) -> AsyncIOMotorDatabase:
        return self.client.get_database('chat_history')