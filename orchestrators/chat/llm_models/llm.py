from typing import TypedDict, Optional, List
from dataclasses import dataclass, field
from langchain_huggingface import HuggingFaceEndpoint

class PromptDict(TypedDict):
    title: str
    prompt: str

class EndpointDict(TypedDict):
    url: str
    type: str

class ParameterDict(TypedDict, total=False):
    max_new_tokens: int
    stop_sequences: List[str]
    streaming: bool
    truncate: int
    do_sample: bool
    repetition_penalty: float
    top_k: int
    top_p: float
    temperature: float
    timeout: int
    task: str

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
    endpoint_object: HuggingFaceEndpoint = field(init=False, repr=False)