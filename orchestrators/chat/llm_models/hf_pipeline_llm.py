from dataclasses import dataclass
from langchain_huggingface import HuggingFacePipeline
from orchestrators.chat.llm_models.llm import LLM

@dataclass(kw_only=True, slots=True)
class HFPipelineLLM(LLM):
    pass