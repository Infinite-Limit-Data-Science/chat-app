from dataclasses import dataclass
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from .base_inference_endpoint import BaseInferenceEndpoint

@dataclass(kw_only=True, slots=True)
class HuggingFaceTextEmbeddingsInference(BaseInferenceEndpoint):
    def __post_init__(self) -> None:
        """
        The name huggingfacehub_api_token is a misnomer because it's actually just a JWT
        but the naming is required by HuggingFaceEndpointEmbeddings
        """
        self._initialize_endpoint_object()

    def _initialize_endpoint_object(self) -> None:
        # TODO: replace HuggingFaceEndpointEmbeddings with my custom implementation at clients.huggingface.huggingface_embeddings
        self.endpoint_object = HuggingFaceEndpointEmbeddings(
            model=self.endpoint['url'],
            task=self.task,
            huggingfacehub_api_token=self.token,
        )

# The base Embeddings class in LangChain provides two methods: one for embedding documents and one for embedding a query. The former takes as input multiple texts, while the latter takes a single text. The reason for having these as two separate methods is that some embedding providers have different embedding methods for documents (to be searched over) vs queries (the search query itself).
# I should have two packages, one that refer to the Embeddings Inference Server (such as HF TGI) and the second package which refers to the types of Embeddings, such as HuggingfaceEmbeddings. Then The Embeddings Server has a property that refers to the embeddings type associated with it!!!!

# Embedding Documents (embed_documents):

# Purpose: This method is used to create embeddings for a collection of texts (e.g., documents or chunks of text). These embeddings are typically stored in a vector database and are later used to find relevant content during a search or retrieval process.

# Embedding Queries (embed_query):
# - Purpose: This method is used to embed a single query. The query embedding is compared to the pre-stored document embeddings to retrieve the most relevant documents.

# Why Are These Separate Methods?

# Some embedding providers (e.g., OpenAI, Cohere, etc.) treat queries and documents differently because they are optimized for different purposes:
# - Document embeddings are designed to capture the content and meaning of a piece of text to enable similarity searches.
# - Query embeddings are designed to represent user search intent, optimized for matching against the stored document embeddings.
# THIS IS IMPORTANT I NEVER THOUGHT OF EMBED QUERY IN THIS WAY!!!!

