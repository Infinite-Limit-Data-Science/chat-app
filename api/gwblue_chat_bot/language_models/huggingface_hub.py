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


class HuggingFaceHub(BaseModel):
    config: ChatBotConfig
    model_types: Optional[Dict[str, Any]] = Field(default_factory={}, exclude=True)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        from ...gwblue_huggingface import (
            HuggingFaceLLM,
            HuggingFaceChatModel,
            HuggingFaceEmbeddings,
        )

        if self.config.llm:
            llm = HuggingFaceLLM(
                base_url=self.config.llm.endpoint,
                credentials=self.config.llm.token,
                provider=self.config.llm.provider,
                model=self.config.llm.model,
                max_tokens=_MAX_TOTAL_TOKENS - _MAX_INPUT_TOKENS,
                temperature=self.config.llm.parameters["temperature"]
                or _DEFAULT_TEMPERATURE,
                logprobs=True,
            )

            self.model_types["chat_model"] = HuggingFaceChatModel(llm=llm)

        if self.config.guardrails:
            llm = HuggingFaceLLM(
                base_url=self.config.guardrails.endpoint,
                credentials=self.config.guardrails.token,
                provider=self.config.guardrails.provider,
                model=self.config.guardrails.model,
                max_tokens=10,
                temperature=0,
            )

            self.model_types["guardrails"] = HuggingFaceChatModel(llm=llm)

        if self.config.embeddings:
            self.model_types["embeddings"] = HuggingFaceEmbeddings(
                model=self.config.embeddings.model,
                base_url=self.config.embeddings.endpoint,
                credentials=self.config.embeddings.token,
                provider=self.config.embeddings.provider,
            )
        return self

    def __call__(self, model_type: str) -> BaseChatModel | Embeddings:
        return self.model_types.get(model_type)
