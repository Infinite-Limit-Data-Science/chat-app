from typing import TypedDict, Optional
from dataclasses import dataclass, field
from langchain_huggingface import HuggingFaceEndpoint

class PromptDict(TypedDict):
    title: str
    prompt: str

class ParameterDict(TypedDict):
    stop: str
    truncate: Optional[str]
    max_new_tokens: int

@dataclass
class HFTGI(frozen=True, kw_only=True, slots=True):
    name: str
    description: str
    default_prompt: PromptDict
    parameters: ParameterDict
    endpoint: HuggingFaceEndpoint = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.endpoint = HuggingFaceEndpoint(**self.parameters)