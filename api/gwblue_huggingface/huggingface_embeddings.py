import os
from typing import Optional, List
from urllib.parse import urlparse
from langchain_core.embeddings import Embeddings
from pydantic import ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self
from .inference_schema import HuggingFaceInferenceServerMixin
from .huggingface_inference_server_config import HuggingFaceEmbeddingsConfig
from .huggingface_inference_client import HuggingFaceInferenceClient

class HuggingFaceBaseEmbeddings(HuggingFaceInferenceServerMixin, Embeddings):
    client: Optional[HuggingFaceInferenceClient] = Field(description='Low-level Inference Client to interface to the self-hosted HF TEI Server', default=None)

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        arbitrary_types_allowed=True,
    )

    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        parsed_url = urlparse(value)
        if parsed_url.path not in ('', '/'):
            raise ValueError(f'Invalid base_url: {value}. Must not contain extra path segments.')
        return value

class HuggingFaceEmbeddings(HuggingFaceBaseEmbeddings):
    # config: Optional[HuggingFaceEmbeddingsConfig] = None

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        # if not self.config:
        #     config = HuggingFaceEmbeddingsConfig(**{
        #         'url': self.base_url,
        #         'auth_token': self.credentials,
        #         'provider': self.provider,
        #         'model': self.model,
        #         'timeout': self.timeout,
        #         'headers': self.headers,
        #     })
        # config = self.config or config
            
        client = HuggingFaceInferenceClient(
            base_url=self.base_url,
            credentials=self.credentials,
            provider=self.provider,
            model=self.model,
            timeout=self.timeout,
            headers=self.headers
        )
        self.client = client

        return self

    def embed_documents(self, texts: List[str], **feat_extract_kwargs) -> List[List[float]]:
        #  Replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        texts = texts[0] if len(texts) == 1 else texts
        embeddings = self.client.feature_extraction(texts, **feat_extract_kwargs).tolist() 
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.embed_documents([text])[0]
        return embedding
    
    async def aembed_documents(self, texts: List[str], **feat_extract_kwargs) -> List[List[float]]:
        #  Replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        texts = texts[0] if len(texts) == 1 else texts
        embeddings = (await self.client.afeature_extraction(texts, **feat_extract_kwargs)).tolist()
        return embeddings
    
    async def aembed_query(self, text: str) -> List[float]:
        embedding = (await self.aembed_documents([text]))[0]
        return embedding