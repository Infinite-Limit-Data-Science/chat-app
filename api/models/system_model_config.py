from datetime import datetime
from typing import List, TypedDict, Optional
from .abstract_model import AbstractModel
from .mongo_schema import ChatSchema, Field, PyObjectId
from .llm_schema import LLMBase


class SystemModelConfig(AbstractModel):
    __modelname__ = "system_model_configs"

    @classmethod
    def get_model_name(cls):
        return cls.__modelname__


class EndpointDict(TypedDict):
    url: str
    provider: str


class SystemModelConfigSchema(LLMBase):
    endpoints: List[EndpointDict] = Field(
        description="Endpoint URL and Provider (e.g. hf-inference, vllm, fireworks-ai, etc)"
    )
    stream: Optional[bool] = Field(description="Enable streaming", default=True)

    class Config:
        arbitrary_types_allowed = True


class SystemModelConfigIdSchema(ChatSchema):
    id: PyObjectId = Field(alias="_id", description="bson object id")


class UpdateSystemModelConfigSchema(LLMBase):
    endpoints: Optional[List[EndpointDict]]
    updatedAt: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True


class SystemModelConfigCollectionSchema(ChatSchema):
    system_model_configs: List[SystemModelConfigSchema]
