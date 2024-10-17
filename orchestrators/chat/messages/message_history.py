import logging
from typing import Sequence, Any
from dataclasses import dataclass, field, asdict
from bson import ObjectId
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables.history import (
    RunnableWithMessageHistory, 
    Runnable, 
    MessagesOrDictWithMessages
)
from orchestrators.chat.messages.my_mongodb_chat_message_history import MyMongoDBChatMessageHistory

@dataclass(kw_only=True, slots=True)
class BaseMessageHistorySchema:
    """Base Schema for a data store like Redis, MongoDB, PostgreSQL, ChromaDB, etc"""
    connection_string: str
    database_name: str
    history_size: int = 100 # restrict message history to 100 messages
    session_id_key: str

@dataclass(kw_only=True, slots=True)
class MongoMessageHistorySchema(BaseMessageHistorySchema):
    """Schema for MongoDB data store"""    
    collection_name: str
    create_index: bool = True
    session_id: ObjectId

class MongoMessageHistory:
    def __init__(self, schema: MongoMessageHistorySchema):
        self._schema = schema
        self.chat_message_history = MyMongoDBChatMessageHistory(**asdict(self._schema))

    @property
    def messages(self) -> list[BaseMessage]:
        return self.chat_message_history.messages
    
    @property
    def has_no_messages(self) -> bool:
        return not self.messages

    def add_messages(self, messages: Sequence[BaseMessage]):
        """Add messages to store"""
        self.chat_message_history.add_messages(messages)

    def system(self, prompt: str, **kwargs: Any) -> SystemMessage:
        """Add system message to store"""
        system_message = SystemMessage(prompt, **kwargs)
        self.add_messages([system_message])
        return system_message

    async def aadd_messages(self, messages: Sequence[BaseMessage]):
        """Add messages to store"""
        await self.chat_message_history.aadd_messages(messages)

    async def asystem(self, prompt: str, **kwargs: Any) -> SystemMessage:
        """Add system message to store"""
        system_message = SystemMessage(prompt, **kwargs)
        await self.aadd_messages([system_message])
        return system_message

    async def ahuman(self, message: dict) -> HumanMessage:
        """Add system message to store"""
        human_message = HumanMessage(**message)
        await self.aadd_messages([human_message])
        return human_message
    
    async def aai(self, message_schema: dict) -> AIMessage:
        """Add ai message to store"""
        ai_message = AIMessage(message_schema)
        await self.aadd_messages([ai_message])
        return ai_message

    async def abulk_add(self, messages: Sequence[BaseMessage]) -> bool:
        """Bulk add operation to store"""
        await self.aadd_messages(messages)
        return True

    def get_session_history(self):
        return self.chat_message_history

    def get(self, chain: Runnable[MessagesOrDictWithMessages, MessagesOrDictWithMessages | str | BaseMessage], rag_chain: bool) -> RunnableWithMessageHistory:
        """Wraps a Runnable with a Chat History Runnable"""
        keys = {
            'input_messages_key': 'input',
            'history_messages_key': 'chat_history'
        }
        if rag_chain:
            keys['output_messages_key'] = 'answer'

        return RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            **keys
        )