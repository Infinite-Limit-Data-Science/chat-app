import pytest
import os
from typing import List
from langchain_core.documents import Document
from ..huggingface_embeddings import HuggingFaceEmbeddings
from ..huggingface_transformer_tokenizers import BgeLargePretrainedTokenizer 
from ...langchain_chunkinator import Chunkinator
from .corpus import dummy_corpus1

from langchain_redis import RedisConfig
from langchain_redis import RedisVectorStore

from langchain_core.example_selectors import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)

attributes = {
    'name': 'BAAI/bge-large-en-v1.5',
    'endpoint': {'url':'http://100.28.34.190:8070/', 'type':'tei'},
    'max_batch_tokens': 32768,
    'max_client_batch_size': 128,
    'max_batch_requests': 64,
    'num_workers': 8,
    'auto_truncate': True,
    'token': 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHAiOiJzdmMtY2hhdC10ZXN0Iiwic3ViIjoibjFtNCIsIm1haWwiOiJqb2huLmRvZUBiY2JzZmwuY29tIiwic3JjIjoiam9obi5kb2VAYmNic2ZsLmNvbSIsInJvbGVzIjpbIiJdLCJpc3MiOiJQTUktVGVzdCIsImF0dHJpYnV0ZXMiOlt7ImdpdmVubmFtZSI6IkpvaG4ifSx7InNuIjoiSkRvZSJ9LHsibWFpbCI6ImpvaG4uZG9lQGJjYnNmbC5jb20ifSx7ImRpc3BsYXluYW1lIjoiSkRvZSwgSm9obiJ9LHsiYmNic2ZsLWlkbVBpY3R1cmVVUkwiOiIifV0sImF1ZCI6ImNoYXRhcHAtdHN0YS50aHJvdGwuY29tIiwiZ2l2ZW5uYW1lIjoiSm9obiIsImRpc3BsYXluYW1lIjoiRG9lLCBKb2huIiwic24iOiJKRG9lIiwiaWRtX3BpY3R1cmVfdXJsIjoiIiwiZXhwIjoxODkzNDU2MDAwLCJpYXQiOjE3MTQxNDQ4NDEsInNlc3Npb25faWQiOiIiLCJqdGkiOiIifQ.rxHyA_WeMprlMtDsTGPvqgjRbQ2qT7VkiT6Ak1aSQmTl3nOFR_v0ev2AmUogUHXJi9CmGZcw3i-Wsis86ggOJKl4e7TwuKSBqt-s81jzGePI2yIsyKInEXwieKHXpWl1JFMtSkDpkRBeaiSlM1qpJ33BJLekRRkW-mDhV-yG5VVxyOWxRZDSfXRgrQ3CoNzChvITqdC1VOCeMAMI5Vg5zvo9bNOjOqOCLEtncsHdDiD7gYmPsGWeR9eXcT0y2-KONa0LvsYBewBcXjvJE63xe3XViiQ3HQPayjA1UAxWekD83_Kq7y-LJEjrQNNphEq_XyocpzvlmK-tlf59UGJJcw'
}

class ConcreteEmbeddingLike:
    @property
    def name(self) -> str:
        return 'BAAI/bge-large-en-v1.5'
    
    @property
    def max_batch_tokens(self) -> int:
        return 32768

    @property
    def max_batch_requests(self) -> int:
        return 64

@pytest.fixture
def chunks() -> List[str]:
    documents = [Document(page_content=dummy_corpus1)]
    chunkinator = Chunkinator.Base(documents, ConcreteEmbeddingLike())
    return chunkinator.chunk() 

@pytest.fixture
def embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        base_url=attributes['endpoint']['url'],
        credentials=attributes['token'],
    )

@pytest.fixture
def tokenizer() -> BgeLargePretrainedTokenizer:
    return BgeLargePretrainedTokenizer()

@pytest.fixture
def vectorstore(
    embeddings: HuggingFaceEmbeddings, 
    tokenizer: BgeLargePretrainedTokenizer
) -> RedisVectorStore:
    # TODO: need to create a cleaner wrapper class for redis vectorstore
    config = RedisConfig(
        index_name="test1",
        redis_url=os.environ['REDIS_URL'],
        metadata_schema=[
            {"name": "source", "type": "tag"},
        ],
        # setting this avoids unnecessary request for embeddings
        embedding_dimensions=tokenizer.dimensions
    )

    return RedisVectorStore(embeddings, config=config)

@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_embed_documents(embeddings: HuggingFaceEmbeddings):
    """
    Embeds text as high-dimensional vectors using the lower level 
    `HuggingFaceInferenceClient` class 
    
    To implement the langchain_core Embeddings abstract class,
    `embed_documents` take the returned numpy float32 arrays
    and converts them to list[list[float]]

    list[float] represents a vector and list[list[float]] 
    represents a list of vectors (in the case of multiple
    text sequences passed in as an array of texts). Each of
    those text sequences will be its own vector.

    Example:
    curl 127.0.0.1:8080/embed \
        -X POST \
        -d '{"inputs":"What is Deep Learning?"}' \
        -H 'Content-Type: application/json'
    curl 127.0.0.1:8080/embed \
        -X POST \
        -d '{"inputs":["Today is a nice day", "I like you"]}' \
        -H 'Content-Type: application/json'    

    The first request will return a single vector. The second
    will return 2 vectors.    
    """
    embedded_vectors = embeddings.embed_documents([dummy_corpus1])
    assert len(embedded_vectors) > 0

@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_embed_multiple_documents(embeddings: HuggingFaceEmbeddings):
    embedded_vectors = embeddings.embed_documents([dummy_corpus1, dummy_corpus1.upper()])
    assert len(embedded_vectors) > 0

@pytest.mark.skip(reason="Temporarily disabled for debugging")    
def test_embed_query(embeddings: HuggingFaceEmbeddings):
    """
    Takes a single query text. In effect, this returns a single
    vector in a list.
    """
    embedded_vectors = embeddings.embed_query(dummy_corpus1)
    assert len(embedded_vectors) > 0

@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_embed_documents_in_vector_db(vectorstore: RedisVectorStore):
    """
    Search over unstructured data embedded in vector database
    
    The returned IDs are just document references in Redis, 
    which combine the index name with a unique key string. The 
    ids are not high-dimensional float32 vectors or tokens part 
    of a model's vocabulary. They are simply internal Redis 
    identifiers
    """
    ids = vectorstore.add_texts([dummy_corpus1], [{'source': 'book'}])
    # TODO: need to delete vectors from vectorstore since this is just testing
    assert ids[0].startswith('test1')

def test_embed_documents_with_similarity_search(vectorstore: RedisVectorStore, chunks: List[str]):
    vectorstore.add_documents(chunks)
    query = "What did King Ulfric Stormborn do in 879"
    results = vectorstore.similarity_search(query, k=2)
    results


            # texts = [doc.page_content for doc in documents]
            # metadatas = [doc.metadata for doc in documents]
            # return self.add_texts(texts, metadatas, **kwargs)

            # embeddings = self._embeddings.embed_documents(texts_list)

            # def embed_documents(self, texts: List[str], **feat_extract_kwargs) -> List[List[float]]:
       
            # texts = [text.replace("\n", " ") for text in texts]
            # texts = texts[0] if len(texts) == 1 else texts
            # embeddings = self.client.feature_extraction(texts, **feat_extract_kwargs).tolist() 
      
        #             request_parameters = provider_helper.prepare_request(
        #     inputs=texts,
        #     parameters={
        #         "normalize": normalize,
        #         "prompt_name": prompt_name,
        #         "truncate": truncate,
        #         "truncation_direction": truncation_direction,
        #     },
        #     headers=self.headers or {},
        #     model=self.base_url,
        #     api_key=self.credentials,
        # )

        # response = self.client._inner_post(request_parameters)

# Similarity Selector
# Max Marginal Relevance (MMR) Selector
# N-gram Overlap Selector    
