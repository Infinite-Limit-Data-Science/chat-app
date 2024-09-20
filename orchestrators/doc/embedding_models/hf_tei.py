import logging
from dataclasses import dataclass
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from orchestrators.doc.embedding_models.embedding import BaseEmbedding

@dataclass(kw_only=True, slots=True)
class HFTEI(BaseEmbedding):
    def __post_init__(self) -> None:
        """The name huggingfacehub_api_token is a misnomer because it's actually just a JWT
        but the naming is required by HuggingFaceEndpointEmbeddings

        The model is actually not a model name only. It can be a custom TEI API, which is
        the exact case here
        """
        endpoint = HuggingFaceEndpointEmbeddings(
            model=self.endpoint['url'],
            task=self.task,
            huggingfacehub_api_token=self.token)
        self.endpoint_object = endpoint