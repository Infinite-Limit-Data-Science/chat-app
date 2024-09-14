
from typing import Sequence
from dataclasses import dataclass, field
from langchain.memory import MongoDBChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

_INPUT_MESSAGES_KEY = 'question'
_HISTORY_MESSAGES_KEY = 'history'

@dataclass(kw_only=True, slots=True)
class BaseMessageHistorySchema:
    """Base Schema for a data store like Redis, MongoDB, PostgreSQL, ChromaDB, etc"""
    connection_string: str
    session_id: str

@dataclass(kw_only=True, slots=True)
class MongoMessageHistorySchema(BaseMessageHistorySchema):
    """Schema for MongoDB data store"""
    database_name: str
    collection_name: str
    create_index: bool = True
    message_history: MongoDBChatMessageHistory = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """connect to mongo message history data store"""
        self.message_history = MongoDBChatMessageHistory(
            connection_string=self.connection_string, 
            session_id=self.session_id,
            database_name=self.database_name,
            collection_name=self.collection_name,
            create_index=self.create_index)        
    
class MongoMessageHistory:
    def __init__(self, schema: MongoMessageHistorySchema):
        self._schema = schema

    @property
    def messages(self) -> list[BaseMessage]:
        return self._schema.message_history.messages
    
    @property
    def has_no_messages(self) -> bool:
        return self._schema.message_history.messages == 0

    async def add_messages(self, messages: Sequence[BaseMessage]):
        """Add messages to store"""
        self._schema.message_history.aadd_messages(messages)

    async def system(self, default_prompt: str) -> SystemMessage:
        """Add system message to store"""
        system_message = SystemMessage(default_prompt)
        await self.add_messages([system_message])
        return system_message

    async def human(self, message_schema: dict) -> HumanMessage:
        """Add system message to store"""
        human_message = HumanMessage(message_schema)
        await self.add_messages([human_message])
        return human_message
    
    async def ai(self, message_schema: dict) -> AIMessage:
        """Add ai message to store"""
        ai_message = AIMessage(message_schema)
        await self.add_messages([ai_message])
        return ai_message

    async def bulk_add(self, messages: Sequence[BaseMessage]) -> bool:
        """Bulk add operation to store"""
        await self.add_messages(messages)
        return True
    
    def update_session(self, session_id: str):
        self._schema.message_history.session_id = session_id
        return self._schema.message_history
    
    def runnable(self, chain) -> RunnableWithMessageHistory:
        """Wraps a Runnable with a Chat History Runnable"""
        """session_id is used to determine whether to create a new message history or load existing"""
        """If message history is stored in the mongo store, it will be loaded here"""
        return RunnableWithMessageHistory(
            chain,
            self.update_session,
            input_messages_key=_INPUT_MESSAGES_KEY,
            history_messages_key=_HISTORY_MESSAGES_KEY,
        )