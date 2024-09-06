from typing import Optional, TypeAlias, TypedDict
from pydantic import BaseModel, Field
from models.mongo_schema import PrimaryKeyMixinSchema, TimestampMixinSchema

LLMSchema: TypeAlias = BaseModel

class PromptDict(TypedDict):
    title: str
    prompt: str

class ParameterDict(TypedDict):
    stop: str
    truncate: Optional[str]
    max_new_tokens: int

class LLMBase(PrimaryKeyMixinSchema, TimestampMixinSchema):
    name: str = Field('Model Name')
    description: str = Field('Description of the Model', default='Description of TGI Model')
    default_prompt: PromptDict = Field('Default prompt (Note users can create custom ones)')
    parameters: ParameterDict = Field('Parameters to pass to model')