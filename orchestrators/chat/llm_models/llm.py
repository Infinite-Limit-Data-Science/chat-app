from typing import TypedDict, Optional, List
from dataclasses import dataclass

class PromptDict(TypedDict):
    title: str
    prompt: str

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

@dataclass
class LLM:
    name: str
    description: str
    default_prompt: PromptDict
    parameters: ParameterDict