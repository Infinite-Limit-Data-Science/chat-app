from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    name: str
    endpoint: str
    token: str
    server: str

class LLMConfig(ModelConfig):
    parameters: Dict[str, Any] = Field(default_factory=dict)

class EmbeddingsConfig(ModelConfig):
    ...

class VectorStoreConfig(BaseModel):
    name: str
    url: str
    collection_name: str
    session_id_key: str

class ChatBotConfig(BaseModel):
    llm: LLMConfig
    retry_llm: LLMConfig = Field(
        description='Secondary LLM for second-opinion tasks', 
        default=None
    )
    embeddings: EmbeddingsConfig
    guardrails: Optional[LLMConfig] = None
    vectorstore: VectorStoreConfig
    metadata: Optional[Dict[Any, str]] = None