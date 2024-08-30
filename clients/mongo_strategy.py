import os
from motor import motor_asyncio
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from clients.database_strategy import DatabaseStrategy

class MongoStrategy(DatabaseStrategy):
    def __init__(self):
        self.client = None

    async def connect(self, url) -> AsyncIOMotorClient:
        """Coroutine to connect to Mongo database"""
        self.client = motor_asyncio.AsyncIOMotorClient(url)
        return self.client

    async def close(self) -> None:
        """Coroutine to close connection to Mongo database"""
        self.client.close()

    def get_database(self) -> AsyncIOMotorDatabase:
        """Return Mongo database"""
        if not os.environ['DATABASE_NAME']:
            raise Exception('Missing `DATABASE_NAME` in environment')
        return self.client.get_database(os.environ['DATABASE_NAME'])
    
mongo_instance = MongoStrategy()