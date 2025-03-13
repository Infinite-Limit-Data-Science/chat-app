from typing import Dict, Any, Optional, TypedDict, List
from pydantic import BaseModel, Field, ConfigDict
from redis.client import Redis
from bson import ObjectId


class ModelConfig(BaseModel):
    model: str
    endpoint: str
    token: str
    provider: str


class LLMConfig(ModelConfig):
    parameters: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingsConfig(ModelConfig):
    max_batch_tokens: int


class MetadataSchema(TypedDict):
    name: str
    type: str


class RedisVectorStoreConfig(BaseModel):
    client: Optional[Redis] = None
    url: Optional[str] = None
    metadata_schema: List[MetadataSchema]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class MongoMessageHistoryConfig(BaseModel):
    name: str
    url: str
    collection_name: str
    session_id_key: str
    session_id: Optional[ObjectId] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class ChatBotConfig(BaseModel):
    llm: LLMConfig
    retry_llm: Optional[LLMConfig] = Field(
        description="Secondary LLM for second-opinion tasks", default=None
    )
    embeddings: EmbeddingsConfig
    guardrails: Optional[LLMConfig] = None
    vectorstore: RedisVectorStoreConfig
    message_history: MongoMessageHistoryConfig
