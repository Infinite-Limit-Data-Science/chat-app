from typing import Self, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from ..chat_bot_config import ChatBotConfig

_DEFAULT_TEMPERATURE = 0.8

# TODO: there should be an api call to get these parameters
# rather than having to rely on app definition in configmap
_MAX_INPUT_TOKENS = 12582

_MAX_TOTAL_TOKENS = 16777

_MAX_BATCH_PREFILL_OPERATION_TOKENS = 12582+50

class HuggingFaceHub(BaseModel):
    config: ChatBotConfig
    model_types: Optional[Dict[str, Any]] = Field(default_factory={}, exclude=True)
    
    @model_validator(mode='after')
    def validate_environment(self) -> Self:
        from ...gwblue_huggingface import (
            HuggingFaceInferenceConfig,
            HuggingFaceEmbeddingsConfig,
            HuggingFaceLLM,
            HuggingFaceChatModel,
            HuggingFaceEmbeddings,
        )

        if self.config.llm:
            tgi_self_hosted_config = HuggingFaceInferenceConfig(
                model=self.config.llm.model,
                url=self.config.llm.endpoint,
                auth_token=self.config.llm.token,
                max_input_tokens=_MAX_INPUT_TOKENS,
                max_total_tokens=_MAX_TOTAL_TOKENS,
                max_batch_prefill_tokens=_MAX_BATCH_PREFILL_OPERATION_TOKENS,
            )
        
            llm = HuggingFaceLLM(
                base_url=tgi_self_hosted_config.url,
                credentials=tgi_self_hosted_config.auth_token,
                inference_config=tgi_self_hosted_config,
                max_tokens=tgi_self_hosted_config.available_generated_tokens,
                temperature=self.config.llm.parameters['temperature'] or _DEFAULT_TEMPERATURE,
                logprobs=True
            )

            self.model_types['chat_model'] = HuggingFaceChatModel(llm=llm)

        if self.config.guardrails:
            tgi_self_hosted_config = HuggingFaceInferenceConfig(
                model=self.config.guardrails.model,
                url=self.config.guardrails.endpoint,
                auth_token=self.config.guardrails.token,
                max_input_tokens=_MAX_INPUT_TOKENS,
                max_total_tokens=_MAX_TOTAL_TOKENS,
                max_batch_prefill_tokens=_MAX_BATCH_PREFILL_OPERATION_TOKENS,
            )

            llm = HuggingFaceLLM(
                base_url=tgi_self_hosted_config.url,
                credentials=tgi_self_hosted_config.auth_token,
                inference_config=tgi_self_hosted_config,
                max_tokens=10,
                temperature=0,
            )

            self.model_types['guardrails'] = HuggingFaceChatModel(llm=llm)

        if self.config.embeddings:
            config = HuggingFaceEmbeddingsConfig(
                model=self.config.embeddings.model,
                url=self.config.embeddings.endpoint,
                auth_token=self.config.embeddings.token,
                provider=self.config.embeddings.provider,
                max_batch_tokens=self.config.embeddings.max_batch_tokens,
            )

            self.model_types['embeddings'] = HuggingFaceEmbeddings(
                config=config
            )
        return self


    def __call__(self, model_type: str) -> BaseChatModel | Embeddings:
        return self.model_types.get(model_type)
