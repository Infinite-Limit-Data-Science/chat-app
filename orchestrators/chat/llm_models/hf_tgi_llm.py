import logging
from dataclasses import dataclass, field
from langchain_huggingface import HuggingFaceEndpoint
from orchestrators.chat.llm_models.llm import LLM, StreamingToClientCallbackHandler

@dataclass(kw_only=True, slots=True)
class HFTGI(LLM):
    def __post_init__(self) -> None:
        self.streaming_handler = StreamingToClientCallbackHandler()
        callbacks = [self.streaming_handler]
        endpoint = HuggingFaceEndpoint(
            streaming=True, 
            callbacks=callbacks, 
            **{'endpoint_url': self.endpoint['url'], **self.parameters, 'server_kwargs': dict(self.server_kwargs)})
        self.endpoint_object = endpoint