from dataclasses import dataclass, field
from langchain_huggingface import HuggingFaceEndpoint
from orchestrators.chat.llm import LLM

@dataclass(frozen=True, kw_only=True, slots=True)
class HFTGI(LLM):
    endpoint: HuggingFaceEndpoint = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.endpoint = HuggingFaceEndpoint(**self.parameters)