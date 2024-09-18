from dataclasses import dataclass, field
from langchain_huggingface import HuggingFaceEndpoint
from orchestrators.chat.llm_models.llm import LLM

@dataclass(kw_only=True, slots=True)
class HFTGI(LLM):
    def __post_init__(self) -> None:
        self.endpoint_object = HuggingFaceEndpoint(**{'endpoint_url': self.endpoint['url'], **self.parameters, 'server_kwargs': dict(self.server_kwargs)})