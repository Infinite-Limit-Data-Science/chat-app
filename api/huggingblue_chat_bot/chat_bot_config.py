from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    name: str
    endpoint: Dict[str, Any]
    token: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class EmbeddingConfig(BaseModel):
    name: str
    endpoint: Dict[str, Any]
    token: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class GuardrailsConfig(BaseModel):
    name: str
    endpoint: Dict[str, Any]
    token: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

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
    embeddings: EmbeddingConfig
    guardrails: Optional[GuardrailsConfig] = None
    human_prompt: str
    vectorstore: VectorStoreConfig
    metadata: Optional[Dict[any, str]] = Field(..., default_factory={})