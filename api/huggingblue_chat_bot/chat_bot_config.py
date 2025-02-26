from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from bson import ObjectId

class ModelConfig(BaseModel):
    name: str
    endpoint: str
    token: str
    server: str

class LLMConfig(ModelConfig):
    parameters: Dict[str, Any] = Field(default_factory=dict)

class EmbeddingsConfig(ModelConfig):
    ...

class RedisVectorStoreConfig(BaseModel):
    url: str
    uuid: str
    session_id_key: str
    session_id: Optional[str] = None
    source: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None
    
class MongoMessageHistoryConfig(BaseModel):
    name: str
    url: str
    collection_name: str
    session_id_key: str
    session_id: Optional[ObjectId] = None

class UserConfig(BaseModel):
    uuid: Optional[str] = None
    session_id_key: Optional[str] = None
    session_id: Optional[ObjectId] = None
    
class ChatBotConfig(BaseModel):
    llm: LLMConfig
    retry_llm: Optional[LLMConfig] = Field(
        description='Secondary LLM for second-opinion tasks', 
        default=None
    )
    embeddings: EmbeddingsConfig
    guardrails: Optional[LLMConfig] = None
    vectorstore: RedisVectorStoreConfig
    message_history: MongoMessageHistoryConfig
    user_config: Optional[UserConfig] = None