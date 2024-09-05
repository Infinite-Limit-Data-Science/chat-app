from typing import Optional, TypedDict
from pydantic import BaseModel, Field

class PromptDict(TypedDict):
    title: str
    prompt: str

class ParameterDict(TypedDict):
    stop: str
    truncate: Optional[str]
    max_new_tokens: int

class TGISchema(BaseModel):
    name: str = Field('Model Name')
    endpoint: str = Field('HF TGI Endpoint')
    endpoint_type: str = Field('Endpoint Type, e.g. TGI, SageMaker, etc')
    description: str = Field('Description of the Model', default='HuggingFace provides open-source models')
    default_prompt: PromptDict = Field('Default prompt (Note users can create custom ones)')
    parameters: ParameterDict = Field('Parameters to pass to model')