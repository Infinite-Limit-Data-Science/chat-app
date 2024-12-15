from typing import TypeAlias, Literal
from enum import Enum
from pydantic import BaseModel, Field
from .mongo_schema import PrimaryKeyMixinSchema, TimestampMixinSchema

EmbeddingSchema: TypeAlias = BaseModel

_TextTask = Literal['feature-extraction']

class TEI_Architecture(Enum):
    MAX_BATCH_TOKENS = 512
    MAX_CLIENT_BATCH_SIZE = 64
    MAX_BATCH_REQUESTS = 5
    NUM_WORKERS = 8
    AUTO_TRUNCATE = 0

class EmbeddingBase(PrimaryKeyMixinSchema, TimestampMixinSchema):
    name: str = Field(description='Embedding Model Name')
    description: str = Field(description='Description of the Embedding Model', default='Description of TEI Model')
    task: _TextTask = Field(description='Specify task for what kind of processing or output to use for embedding model', default='feature-extraction')
    dimensions: int = Field(description='In the statistical sense, the number of features embodying a vector and conveyed as numerical vector representations of data')
    max_batch_tokens: int = Field(description='Total number of tokens that can be processed in a single batch', default=TEI_Architecture.MAX_BATCH_TOKENS.value)
    max_client_batch_size: int = Field(description='Limits the maximum number of inputs a single client can send in one request', default=TEI_Architecture.MAX_CLIENT_BATCH_SIZE.value)
    max_batch_requests: int = Field(description='Limits the maximum number of individual requests that can be combined into a single batch for processing.', default=TEI_Architecture.MAX_BATCH_REQUESTS.value)
    num_workers: int = Field(description='Controls the number of tokenizer workers used for payload tokenization, validation and truncation.', default=TEI_Architecture.NUM_WORKERS.value)
    auto_truncate: bool = Field(description='Truncate tokens above max_batch_tokens limit', default=bool(TEI_Architecture.AUTO_TRUNCATE.value))
    active: bool = Field(description='Specify if the model is active', default=False)

    class Config:
        from_attributes = True