from typing import Annotated, Optional, Dict, Any
from typing_extensions import Doc
import ipaddress
from urllib.parse import urlparse
from pydantic import Field, field_validator, model_validator, computed_field
from .inference_schema import InferenceSchema

"""
None of these classes are actually used in the application, but rather printed
during execution of __main__ package through the command-line
"""

class HuggingFaceSelfHostedServer(InferenceSchema):
    model: str = Field(description='Name of model hosted on self-hosted TEI', pattern=r".+/.+")
    url: str = Field(description='The URL of the self-hosted TEI Server')
    provider: str = Field(description='HuggingFace Hub provider, such as hf-inference, vllm, fireworks-ai, etc')
    auth_token: Optional[str] = Field(description='Auth token to access self-hosted server', default=None)

    @field_validator('url')
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        parsed_url = urlparse(value)
        
        if not parsed_url.hostname:
            raise ValueError(f'Invalid URL: "{value}". Hostname is required.')
        
        if not value.endswith('/'):
            raise ValueError(f'URL must end with a forward slash: "{value}"')
        
        try:
            ipaddress.ip_address(parsed_url.hostname)
            if parsed_url.path not in ('', '/'):
                raise ValueError(
                    f'Invalid base_url: "{value}". Must not contain extra path segments for IP-based URL.'
                )
        except ValueError:
            pass

        return value
        
    @property
    def self_hosted() -> bool:
        return True

class HuggingFaceInferenceConfig(HuggingFaceSelfHostedServer):
    max_input_tokens: Optional[int] = Field(description='Maximum number of input tokens allowed for each individual request; that is, it sets the maximum number of tokens allowed in the input prompt for a request.', default=None)
    max_total_tokens: Optional[int] = Field(description='Sets the maximum total number of tokens (input tokens + generated tokens) allowed for a request.', default=None)
    max_batch_prefill_tokens: Optional[int] = Field(description='Limit on the number of tokens for the prefill operation in a batch. Since prefill phase is a crucial part of text generation in which the model processes the initial tokens before producing new tokens for each request, it take the most memory and is compute bound. It default to `max_input_tokens (the initial tokens) + 50` to give a bit of room. Note since a batch can consists of multiple queries in a single request, this limit applies to all tokens in all queries of the batch.', default=None)
    max_batch_total_tokens: Optional[int] = Field(description='Hard limit on the total number of tokens (input + generated tokens) across all queries in a batch.', default=None)

    @model_validator(mode='before')
    @classmethod
    def check_token_boundaries(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        max_input_tokens = values.get('max_input_tokens')
        max_total_tokens = values.get('max_total_tokens')
        max_batch_prefill_tokens = values.get('max_batch_prefill_tokens')
        max_batch_total_tokens = values.get('max_batch_total_tokens')

        if max_input_tokens is not None and max_total_tokens is not None:
            if max_input_tokens > max_total_tokens:
                raise ValueError(
                    "`max_input_tokens` cannot be greater than `max_total_tokens`."
                )

        if max_batch_prefill_tokens is not None and max_batch_total_tokens is not None:
            if max_batch_prefill_tokens > max_batch_total_tokens:
                raise ValueError(
                    "`max_batch_prefill_tokens` cannot be greater than `max_batch_total_tokens`."
                )

        return values

    @computed_field
    @property
    def available_generated_tokens(self) -> Optional[int]:
        if self.max_total_tokens is not None and self.max_input_tokens is not None:
            return max(0, self.max_total_tokens - self.max_input_tokens)
        return None

class HuggingFaceEmbeddingsConfig(HuggingFaceSelfHostedServer):
    max_batch_tokens: Optional[int] = Field(description='Total number of tokens that can be processed in a single batch. A batch refers to a list of queries in a single POST request to the TEI server. If max_batch_tokens == 1000, you could send 10 queries each containing 100 tokens in one request. OR send one query containing 1000 tokens in one request.', default=None)