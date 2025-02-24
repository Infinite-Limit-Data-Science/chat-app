import os
from typing import Dict
from pydantic import BaseModel, model_validator
from langchain_core.language_models.chat_models import BaseChatModel

class HuggingFaceInference(BaseModel):
    chat_model: BaseChatModel = None
    safety_model: BaseChatModel = None
    
    @model_validator(mode='before')
    @classmethod
    def validate_environment(cls, values: Dict[str, any]) -> Dict[str, any]:
        from ...huggingblue_inference_kit import (
            HuggingFaceTGIConfig,
            HuggingFaceLLM,
            HuggingFaceChatModel, 
        )

        tgi_self_hosted_config = HuggingFaceTGIConfig(
            name=os.environ['GUARDRAILS_MODEL'],
            url=os.environ['GUARDRAILS_TGI_URL'],
            auth_token=os.environ['GUARDRAILS_AUTH'],
            max_input_tokens=os.getenv('GUARDRAILS_MAX_INPUT_TOKENS', 12582),
            max_total_tokens=os.getenv('GUARDRAILS_MAX_TOTAL_TOKENS', 16777),
            max_batch_prefill_tokens=os.getenv('GUARDRAILS_MAX_PREFILL_TOKENS', 5_000_000),
            payload_limit=5_000_000
        )
     
        if not values['base_model']:
            llm = HuggingFaceLLM(
                base_url=tgi_self_hosted_config.url,
                credentials=tgi_self_hosted_config.auth_token,
                tgi_config=tgi_self_hosted_config,
                max_tokens=tgi_self_hosted_config.available_generated_tokens,
                temperature=0.8,
                logprobs=True
            )
            values['chat_model'] = HuggingFaceChatModel(llm=llm)

        if not values['guardrails']:
            guardrails = HuggingFaceLLM(
                base_url=tgi_self_hosted_config.url,
                credentials=tgi_self_hosted_config.auth_token,
                tgi_config=tgi_self_hosted_config,
                max_tokens=50,
                temperature=0
            )
            values['safety_model'] = HuggingFaceChatModel(llm=guardrails)

        return values