from typing import TypedDict
from dataclasses import dataclass, field
from langchain_huggingface import HuggingFaceEndpoint
from orchestrators.chat.llm_models.llm import LLM

class EndpointDict(TypedDict):
    url: str
    type: str

@dataclass(frozen=True, kw_only=True, slots=True)
class HFTGI(LLM):
    endpoint: EndpointDict
    endpoint_object: HuggingFaceEndpoint = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.endpoint_object = HuggingFaceEndpoint({'endpoint_url': self.endpoint.url, **self.parameters})