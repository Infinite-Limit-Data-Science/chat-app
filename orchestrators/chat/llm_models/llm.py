import logging
from typing import TypedDict, Optional, List, Any
from dataclasses import dataclass, field
from langchain_core.outputs.llm_result import LLMResult
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.callbacks.base import BaseCallbackHandler
import asyncio

class PromptDict(TypedDict):
    title: str
    prompt: str

class EndpointDict(TypedDict):
    url: str
    type: str

class ParameterDict(TypedDict, total=False):
    max_new_tokens: int
    stop_sequences: List[str]
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

class StreamingToClientCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await self.queue.put(token)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        await self.queue.put(None)

    async def get_streamed_response(self):
        while True:
            token = await self.queue.get()
            if token is None:
                break
            yield token
        yield ''

@dataclass
class LLM:
    name: str
    description: str
    preprompt: str
    parameters: ParameterDict
    server_kwargs: ServerKwargDict
    endpoint: EndpointDict = field(default=None)
    endpoint_object: HuggingFaceEndpoint = field(init=False, repr=False)
    streaming_handler: StreamingToClientCallbackHandler = field(init=False, repr=False)