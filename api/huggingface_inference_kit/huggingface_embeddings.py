from typing import Optional, List
from urllib.parse import urlparse
from langchain_core.embeddings import Embeddings
from pydantic import ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self
from .inference_schema import HuggingFaceTEIMixin
from .huggingface_inference_client import HuggingFaceInferenceLike, HuggingFaceInferenceClient

class HuggingFaceBaseEmbeddings(HuggingFaceTEIMixin, Embeddings):
    client: Optional[HuggingFaceInferenceLike] = Field(description='Low-level Inference Client to interface to the self-hosted HF TEI Server', default=None)

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
    """
    This is a drop-in replacement of the langchain_huggingface `HuggingFaceEndpointEmbeddings` class
    which is broken, due to the fact that it relies on a deprecated method `post` of the inference client

    Drop-in replacement to:
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    """
    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        client = HuggingFaceInferenceClient(
            base_url=self.base_url,
            credentials=self.credentials,
            timeout=self.timeout,
            headers=self.headers
        )
        self.client = client

        return self    

    def embed_documents(self, texts: List[str], **feat_extract_kwargs) -> List[List[float]]:
        """
        Embeds text as high-dimensional vectors
        TEI supports both single input and multiple inputs in a batch:
        curl 127.0.0.1:8080/embed \
            -X POST \
            -d '{"inputs":"What is Deep Learning?"}' \
            -H 'Content-Type: application/json'
        curl 127.0.0.1:8080/embed \
            -X POST \
            -d '{"inputs":["Today is a nice day", "I like you"]}' \
            -H 'Content-Type: application/json'
        
        Replace newlines, which can negatively affect performance.
        """
        texts = [text.replace("\n", " ") for text in texts]
        texts = texts[0] if len(texts) == 1 else texts
        embeddings = self.client.feature_extraction(texts, **feat_extract_kwargs).tolist() 
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.embed_documents([text])[0]
        return embedding