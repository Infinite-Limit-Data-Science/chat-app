import logging
import os
from motor import motor_asyncio
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from clients.database_strategy import DatabaseStrategy

logging.getLogger('pymongo').setLevel(logging.WARNING)

_MESSAGE_HISTORY_COLLECTION = 'messages'
_MESSAGE_HISTORY_KEY = 'conversation_id'

class MongoStrategy(DatabaseStrategy):
    def __init__(self, url: str):
        self._client = None
        self._database_name = os.environ['DATABASE_NAME']
        self._message_history_collection = _MESSAGE_HISTORY_COLLECTION
        self._message_history_key = _MESSAGE_HISTORY_KEY
        self._url = url

    async def connect(self) -> AsyncIOMotorClient:
        """Coroutine to connect to Mongo database"""
        self._client = motor_asyncio.AsyncIOMotorClient(self._url)
        return self._client

    async def close(self) -> None:
        """Coroutine to close connection to Mongo database"""
        self._client.close()

    def get_database(self) -> AsyncIOMotorDatabase:
        """Return Mongo database"""
        return self._client.get_database(self._database_name)
    
    @property
    def name(self):
        return self._database_name
    
    @property
    def message_history_collection(self):
        return self._message_history_collection
    
    @property
    def message_history_key(self):
        return self._message_history_key

    @property
    def connection_string(self):
        return self._url

if not os.environ['DATABASE_NAME']:
    raise Exception('Missing `DATABASE_NAME` in environment, therefore, not trying to connect')
mongo_instance = MongoStrategy(os.environ['MONGODB_URL'])