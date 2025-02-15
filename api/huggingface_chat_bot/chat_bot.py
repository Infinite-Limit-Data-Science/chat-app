from typing import Generic, TypeVar, List, Union
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage

LanguageModelOutput = Union[BaseMessage, str]

I = TypeVar(bound=BaseChatModel)
O = TypeVar(bound=List[LanguageModelOutput])

class HuggingFaceChatBot(Generic[I, O], RunnableSequence):
    pass