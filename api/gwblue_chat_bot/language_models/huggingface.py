from typing import Self, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from ..chat_bot_config import ChatBotConfig

# TODO: there should be an api call to get these parameters
# rather than having to rely on app definition in configmap
_DEFAULT_TEMPERATURE = 0.8

_MAX_INPUT_TOKENS = 12582

_MAX_TOTAL_TOKENS = 16777

_MAX_BATCH_PREFILL_OPERATION_TOKENS = 12582+50

_PAYLOAD_LIMIT = 5_000_000

_EMBEDDINGS_MAX_BATCH_TOKENS = 32768

_EMBEDDINGS_MAX_CLIENT_BATCH_SIZE = 128

_EMBEDDINGS_MAX_BATCH_REQUESTS = 64

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
                max_input_tokens=_MAX_INPUT_TOKENS,
                max_total_tokens=_MAX_TOTAL_TOKENS,
                max_batch_prefill_tokens=_MAX_BATCH_PREFILL_OPERATION_TOKENS,
                payload_limit=_PAYLOAD_LIMIT
            )
        
            llm = HuggingFaceLLM(
                base_url=tgi_self_hosted_config.url,
                credentials=tgi_self_hosted_config.auth_token,
                tgi_config=tgi_self_hosted_config,
                max_tokens=tgi_self_hosted_config.available_generated_tokens,
                temperature=self.config.llm.parameters['temperature'] or _DEFAULT_TEMPERATURE,
                logprobs=True
            )

            self.model_types['chat_model'] = HuggingFaceChatModel(llm=llm)

        if self.config.guardrails:
            tgi_self_hosted_config = HuggingFaceTGIConfig(
                name=self.config.guardrails.name,
                url=self.config.guardrails.endpoint,
                auth_token=self.config.guardrails.token,
                max_input_tokens=_MAX_INPUT_TOKENS,
                max_total_tokens=_MAX_TOTAL_TOKENS,
                max_batch_prefill_tokens=_MAX_BATCH_PREFILL_OPERATION_TOKENS,
                payload_limit=_PAYLOAD_LIMIT
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
                max_batch_tokens=_EMBEDDINGS_MAX_BATCH_TOKENS,
                max_client_batch_size=_EMBEDDINGS_MAX_CLIENT_BATCH_SIZE,
                max_batch_requests=_EMBEDDINGS_MAX_BATCH_REQUESTS,
                auto_truncate=True
            )

            self.model_types['embeddings'] = HuggingFaceEmbeddings(
                base_url=tei_self_hosted_config.url,
                credentials=tei_self_hosted_config.auth_token
            )
        return self


    def __call__(self, model_type: str) -> BaseChatModel | Embeddings:
        return self.model_types.get(model_type)
