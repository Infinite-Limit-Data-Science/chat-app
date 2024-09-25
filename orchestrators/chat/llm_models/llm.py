import logging
from typing import TypedDict, List
from dataclasses import dataclass, field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace
from orchestrators.chat.llm_models.my_chat_huggingface import MyChatHuggingFace

class PromptDict(TypedDict):
    title: str
    prompt: str

class EndpointDict(TypedDict):
    url: str
    type: str

class ParameterDict(TypedDict, total=False):
    max_new_tokens: int
    truncate: int
    do_sample: bool
    repetition_penalty: float
    top_k: int
    top_p: float
    temperature: float
    # stop sequences is necessary for a component implementing the LLM Runnable
    stop_sequences: List[str]

class ServerHeaderDict(TypedDict):
    Authorization: str

class ServerKwargDict(TypedDict):
    headers: ServerHeaderDict

@dataclass
class LLM:
    name: str
    description: str
    preprompt: str
    parameters: ParameterDict
    server_kwargs: ServerKwargDict
    endpoint: EndpointDict = field(default=None)
    # endpoint_object must implement BaseChatModel (integrated with Runnable and SequenceRunnable)
    endpoint_object: MyChatHuggingFace | ChatHuggingFace = field(init=False, repr=False)