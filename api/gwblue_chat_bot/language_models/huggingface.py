import os
from typing import Self, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from ..chat_bot_config import ChatBotConfig

class HuggingFaceInference(BaseModel):
    config: ChatBotConfig
    model_types: Optional[Dict[str, Any]] = Field(default_factory={}, exclude=True)
    
    @model_validator(mode='after')
    def validate_environment(self) -> Self:
        from ...gwblue_huggingface import (
            HuggingFaceTGIConfig,
            HuggingFaceTEIConfig,
            HuggingFaceLLM,
            HuggingFaceChatModel,
            HuggingFaceEmbeddings,
        )

        if self.config.llm:
            tgi_self_hosted_config = HuggingFaceTGIConfig(
                name=self.config.llm.name,
                url=self.config.llm.endpoint,
                auth_token=self.config.llm.token,
                max_input_tokens=12582,
                max_total_tokens=16777,
                max_batch_prefill_tokens=12582+50,
                payload_limit=5_000_000
            )
        
            llm = HuggingFaceLLM(
                base_url=tgi_self_hosted_config.url,
                credentials=tgi_self_hosted_config.auth_token,
                tgi_config=tgi_self_hosted_config,
                max_tokens=tgi_self_hosted_config.available_generated_tokens,
                temperature=0.8,
                logprobs=True
            )

            self.model_types['chat_model'] = HuggingFaceChatModel(llm=llm)

        if self.config.guardrails:
            tgi_self_hosted_config = HuggingFaceTGIConfig(
                name=self.config.guardrails.name,
                url=self.config.guardrails.endpoint,
                auth_token=self.config.guardrails.token,
                max_input_tokens=12582,
                max_total_tokens=16777,
                max_batch_prefill_tokens=12582+50,
                payload_limit=5_000_000
            )

            llm = HuggingFaceLLM(
                base_url=tgi_self_hosted_config.url,
                credentials=tgi_self_hosted_config.auth_token,
                tgi_config=tgi_self_hosted_config,
                max_tokens=10,
                temperature=0,
            )

            self.model_types['guardrails'] = HuggingFaceChatModel(llm=llm)

        if self.config.embeddings:
            tei_self_hosted_config = HuggingFaceTEIConfig(
                name=self.config.embeddings.name,
                url=self.config.embeddings.endpoint,
                auth_token=self.config.embeddings.token,
                max_batch_tokens=32768,
                max_client_batch_size=128,
                max_batch_requests=64,
                auto_truncate=True
            )

            self.model_types['embeddings'] = HuggingFaceEmbeddings(
                base_url=tei_self_hosted_config.url,
                credentials=tei_self_hosted_config.auth_token
            )
        return self


    def __call__(self, model_type: str) -> BaseChatModel | Embeddings:
        return self.model_types.get(model_type)
