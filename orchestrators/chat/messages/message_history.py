from dataclasses import dataclass, field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

@dataclass(frozen=True, kw_only=True, slots=True)
class MessageHistorySchema:
    conversation_id: str
    content: str
    modelDetail: str
    files: list[str] = field(default_factory=list)
    message_history = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Fetch message history for conversation"""
        self.message_history = ''

class AzureTableChatMessageHistory(BaseChatMessageHistory):
    def __init__(
          self,
          session_id: str,
          connection_string: str,
          table_name: str,
          key_prefix: str = "chat_history:",
          ttl: Optional[int] = None        
      ):

    @property
    def key(self) -> str:
        """Construct the record key to use"""
        return self.key_prefix + self.session_id

# class MessageHistory:
#     def __init__(self, schema: MessageHistorySchema):
#         self.schema = schema

#     def session()

#     def system(self) -> SystemMessage:
#         return 

#     def invoke(self) -> HumanMessage:
#         return HumanMessage(content=self.schema.content)