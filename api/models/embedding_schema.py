from typing import TypeAlias
from enum import Enum
from pydantic import BaseModel, Field
from .mongo_schema import PrimaryKeyMixinSchema, TimestampMixinSchema

EmbeddingSchema: TypeAlias = BaseModel

class TEI_ENGINE_ARGS(Enum):
    MAX_BATCH_TOKENS = 24256
    MAX_CLIENT_BATCH_SIZE = 64
    MAX_BATCH_REQUESTS = 5

class EmbeddingBase(PrimaryKeyMixinSchema, TimestampMixinSchema):
    name: str = Field(description='Embedding Model Name')
    max_batch_tokens: int = Field(description='Total number of tokens that can be processed in a single batch', default=TEI_ENGINE_ARGS.MAX_BATCH_TOKENS.value)
    max_client_batch_size: int = Field(description='Limits the maximum number of inputs a single client can send in one request', default=TEI_ENGINE_ARGS.MAX_CLIENT_BATCH_SIZE.value)
    max_batch_requests: int = Field(description='Limits the maximum number of individual requests that can be combined into a single batch for processing.', default=TEI_ENGINE_ARGS.MAX_BATCH_REQUESTS.value)
    provider: str = Field(description='The platform using HuggingFace Hub, e.g. hf-inference, vllm, fireworks-ai, together, etc', default='hf-inference')
    active: bool = Field(description='Specify if the model is active', default=False)

    class Config:
        from_attributes = True