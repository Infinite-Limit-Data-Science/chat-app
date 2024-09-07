from typing import TypedDict, Optional
from dataclasses import dataclass

class PromptDict(TypedDict):
    title: str
    prompt: str

class ParameterDict(TypedDict):
    stop: str
    truncate: Optional[str]
    max_new_tokens: int
    
@dataclass
class LLM:
    name: str
    description: str
    default_prompt: PromptDict
    parameters: ParameterDict