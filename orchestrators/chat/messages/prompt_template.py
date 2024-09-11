from typing import TypedDict
from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

@dataclass
class PromptDict(TypedDict):
    title: str
    content: str

class PromptTemplate:
    def __init__(self, prompt: PromptDict):
        self._prompt = prompt

    def runnable(self):
        """Runnable with placeholder to reference message history"""
        return ChatPromptTemplate.from_messages(
            [
                ("system", self._prompt.content),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )