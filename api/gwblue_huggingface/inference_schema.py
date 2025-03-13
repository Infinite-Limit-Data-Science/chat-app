from typing import TypeAlias, Optional, Dict
from pydantic import BaseModel, Field

InferenceSchema: TypeAlias = BaseModel


class HuggingFaceInferenceServerMixin(InferenceSchema):
    base_url: str = Field(
        description="The base url of the self-hosted Inference Endpoint. Do not specify URL segments like /embed or /chat/completions."
    )
    credentials: str = Field(
        description="The authentication token, can be a bearer token or any other form of token represented as a string type."
    )
    timeout: Optional[float] = Field(
        description="Maximum number of seconds to wait for a response. Default of None means it will wait until server available",
        default=None,
    )
    headers: Optional[Dict[str, str]] = Field(
        description="Additional headers to send to the server. By default only the authorization and user-agent headers are sent.",
        default=None,
    )
    provider: Optional[str] = Field(
        description="Optional driver for inference backend, e.g. how to build requests and parse responses",
        default="hf-inference",
    )
    model: Optional[str] = Field(
        description="Optional model name, not used unless for edge cases", default=None
    )
