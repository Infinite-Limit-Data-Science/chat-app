
from dataclasses import dataclass, field
from langchain.memory import MongoDBChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

@dataclass(kw_only=True, slots=True)
class BaseMessageHistorySchema:
    """Base Schema for a data store like Redis, MongoDB, PostgreSQL, ChromaDB, etc"""
    connection_string: str
    session_id: str

@dataclass(frozen=True, kw_only=True, slots=True)
class MongoMessageHistorySchema(BaseMessageHistorySchema):
    """Schema for MongoDB data store"""
    database_name: str
    collection_name: str
    create_index: bool = True
    message_history = field(init=False, repr=False)

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
        self._new_messages = []

    @property
    def messages(self) -> list[BaseMessage]:
        return self._schema.message_history.messages
    
    @property
    def has_no_messages(self) -> bool:
        return self._schema.message_history.messages == 0

    @messages.setter
    def messages(self, messages=list[BaseMessage]):
        self._new_messages.extend(messages)

    def system(self, default_prompt: str) -> SystemMessage:
        return SystemMessage(default_prompt)

    def human(self, message_schema: dict) -> HumanMessage:
        return HumanMessage(message_schema)
    
    def ai(self, message_schema: dict) -> AIMessage:
        return AIMessage(message_schema)

    async def save(self) -> bool:
        await self._schema.message_history.aadd_messages(self._new_messages)
        self._new_messages = []
        return True