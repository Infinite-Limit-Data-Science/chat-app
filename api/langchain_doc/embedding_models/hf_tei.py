from dataclasses import dataclass
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from .embedding import BaseEmbedding

@dataclass(kw_only=True, slots=True)
class HFTEI(BaseEmbedding):
    def __post_init__(self) -> None:
        """
        The name huggingfacehub_api_token is a misnomer because it's actually just a JWT
        but the naming is required by HuggingFaceEndpointEmbeddings
        """
        self._initialize_endpoint_object()

    def _initialize_endpoint_object(self) -> None:
        self.endpoint_object = HuggingFaceEndpointEmbeddings(
            model=self.endpoint['url'],
            task=self.task,
            huggingfacehub_api_token=self.token,
        )