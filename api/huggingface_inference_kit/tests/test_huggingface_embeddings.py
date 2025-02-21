import pytest
import os
from typing import List, Iterator
from langchain_core.documents import Document
from langchain_redis import RedisConfig
from langchain_redis import RedisVectorStore
from redisvl.query.filter import Tag
from ..huggingface_embeddings import HuggingFaceEmbeddings
from ..huggingface_transformer_tokenizers import BgeLargePretrainedTokenizer 
from ...langchain_chunkinator import Chunkinator
from ..huggingface_inference_server_config import HuggingFaceTEIConfig
from .corpus import dummy_corpus1

@pytest.fixture
def tei_self_hosted_config() -> HuggingFaceTEIConfig:
    return HuggingFaceTEIConfig(
        name='BAAI/bge-large-en-v1.5',
        url='http://100.28.34.190:8070/',
        auth_token='eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHAiOiJzdmMtY2hhdC10ZXN0Iiwic3ViIjoibjFtNCIsIm1haWwiOiJqb2huLmRvZUBiY2JzZmwuY29tIiwic3JjIjoiam9obi5kb2VAYmNic2ZsLmNvbSIsInJvbGVzIjpbIiJdLCJpc3MiOiJQTUktVGVzdCIsImF0dHJpYnV0ZXMiOlt7ImdpdmVubmFtZSI6IkpvaG4ifSx7InNuIjoiSkRvZSJ9LHsibWFpbCI6ImpvaG4uZG9lQGJjYnNmbC5jb20ifSx7ImRpc3BsYXluYW1lIjoiSkRvZSwgSm9obiJ9LHsiYmNic2ZsLWlkbVBpY3R1cmVVUkwiOiIifV0sImF1ZCI6ImNoYXRhcHAtdHN0YS50aHJvdGwuY29tIiwiZ2l2ZW5uYW1lIjoiSm9obiIsImRpc3BsYXluYW1lIjoiRG9lLCBKb2huIiwic24iOiJKRG9lIiwiaWRtX3BpY3R1cmVfdXJsIjoiIiwiZXhwIjoxODkzNDU2MDAwLCJpYXQiOjE3MTQxNDQ4NDEsInNlc3Npb25faWQiOiIiLCJqdGkiOiIifQ.rxHyA_WeMprlMtDsTGPvqgjRbQ2qT7VkiT6Ak1aSQmTl3nOFR_v0ev2AmUogUHXJi9CmGZcw3i-Wsis86ggOJKl4e7TwuKSBqt-s81jzGePI2yIsyKInEXwieKHXpWl1JFMtSkDpkRBeaiSlM1qpJ33BJLekRRkW-mDhV-yG5VVxyOWxRZDSfXRgrQ3CoNzChvITqdC1VOCeMAMI5Vg5zvo9bNOjOqOCLEtncsHdDiD7gYmPsGWeR9eXcT0y2-KONa0LvsYBewBcXjvJE63xe3XViiQ3HQPayjA1UAxWekD83_Kq7y-LJEjrQNNphEq_XyocpzvlmK-tlf59UGJJcw',        
        max_batch_tokens=32768,
        max_client_batch_size=128,
        max_batch_requests=64,
        auto_truncate=True
    )

@pytest.fixture
def chunks(tei_self_hosted_config: HuggingFaceTEIConfig) -> List[str]:
    documents = [Document(page_content=dummy_corpus1, metadata={'source': 'book'})]
    chunkinator = Chunkinator.Base(documents, tei_self_hosted_config)
    return chunkinator.chunk() 

@pytest.fixture
def embeddings(tei_self_hosted_config: HuggingFaceTEIConfig) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        base_url=tei_self_hosted_config.url,
        credentials=tei_self_hosted_config.auth_token
    )

@pytest.fixture
def tokenizer() -> BgeLargePretrainedTokenizer:
    return BgeLargePretrainedTokenizer()

@pytest.fixture
def vectorstore(
    embeddings: HuggingFaceEmbeddings, 
    tokenizer: BgeLargePretrainedTokenizer
) -> Iterator[RedisVectorStore]:
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

    store = RedisVectorStore(embeddings, config=config)

    yield store

    store.index.clear()
    store.index.delete(drop=True)

# @pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_embed_documents(embeddings: HuggingFaceEmbeddings):
    embedded_vectors = embeddings.embed_documents([dummy_corpus1])
    assert len(embedded_vectors) > 0

def test_embed_multiple_documents(embeddings: HuggingFaceEmbeddings):
    embedded_vectors = embeddings.embed_documents([dummy_corpus1, dummy_corpus1.upper()])
    assert len(embedded_vectors) > 0

def test_embed_query(embeddings: HuggingFaceEmbeddings):
    embedded_vectors = embeddings.embed_query(dummy_corpus1)
    assert len(embedded_vectors) > 0

def test_embed_documents_in_vector_db(vectorstore: RedisVectorStore):
    ids = vectorstore.add_texts([dummy_corpus1], [{'source': 'book'}])
    assert ids[0].startswith('test1')

def test_embed_documents_with_similarity_search(vectorstore: RedisVectorStore, chunks: List[str]):
    vectorstore.add_documents(chunks)
    query = "What did King Ulfric Stormborn do in 879"
    results = vectorstore.similarity_search(
        query, 
        k=2, 
        filter=Tag('source') == 'book'
    )
    assert len(results) == 2

def test_embed_documents_with_similarity_search_with_score(vectorstore: RedisVectorStore, chunks: List[str]): 
    vectorstore.add_documents(chunks)
    query = "What did King Ulfric Stormborn do in 879"    
    results = vectorstore.similarity_search_with_score(
        query, 
        k=2, 
        filter=Tag('source') == 'book'
    )

    score1 = results[0][1]
    score2 = results[1][1]

    assert score1 < score2

def test_embed_documents_with_max_marginal_relevance_search(vectorstore: RedisVectorStore, chunks: List[str]):
    vectorstore.add_documents(chunks)
    query = "What did King Ulfric Stormborn do in 879"    
    results = vectorstore.max_marginal_relevance_search(
        query, 
        k=2, 
        filter=Tag('source') == 'book'
    )

    assert len(results) == 2

def test_embed_documents_with_similarity_search_by_vector(embeddings: HuggingFaceEmbeddings, vectorstore: RedisVectorStore, chunks: List[str]):
    vectorstore.add_documents(chunks)
    
    query = "What did King Ulfric Stormborn do in 879"
    float_32_1024_dimensional_bge_vector = embeddings.embed_documents([query])[0]
    results = vectorstore.similarity_search_by_vector(
        float_32_1024_dimensional_bge_vector, 
        k=2, 
        filter=Tag('source') == 'book'
    )
    assert len(results) == 2

def test_embed_documents_with_similarity_search_with_score_by_vector(embeddings: HuggingFaceEmbeddings, vectorstore: RedisVectorStore, chunks: List[str]):
    vectorstore.add_documents(chunks)

    query = "What did King Ulfric Stormborn do in 879"
    float_32_1024_dimensional_bge_vector = embeddings.embed_documents([query])[0]
    results = vectorstore.similarity_search_with_score_by_vector(
        float_32_1024_dimensional_bge_vector, 
        k=2, 
        filter=Tag('source') == 'book'
    )

    score1 = results[0][1]
    score2 = results[1][1]

    assert score1 < score2

def test_embed_documents_with_max_marginal_relevance_search_by_vector(embeddings: HuggingFaceEmbeddings, vectorstore: RedisVectorStore, chunks: List[str]):
    vectorstore.add_documents(chunks)

    query = "What did King Ulfric Stormborn do in 879"
    float_32_1024_dimensional_bge_vector = embeddings.embed_documents([query])[0]
    results = vectorstore.max_marginal_relevance_search_by_vector(
        float_32_1024_dimensional_bge_vector, 
        k=2, 
        filter=Tag('source') == 'book'
    )

    assert len(results) == 2

@pytest.mark.asyncio
async def test_aembed_documents(embeddings: HuggingFaceEmbeddings):
    embedded_vectors = await embeddings.aembed_documents([dummy_corpus1])
    assert len(embedded_vectors) > 0

@pytest.mark.asyncio
async def test_aembed_multiple_documents(embeddings: HuggingFaceEmbeddings):
    embedded_vectors = await embeddings.aembed_documents([dummy_corpus1, dummy_corpus1.upper()])
    assert len(embedded_vectors) > 0

@pytest.mark.asyncio
async def test_aembed_query(embeddings: HuggingFaceEmbeddings):
    embedded_vectors = await embeddings.aembed_query(dummy_corpus1)
    assert len(embedded_vectors) > 0

@pytest.mark.asyncio
async def test_aembed_documents_in_vector_db(vectorstore: RedisVectorStore):
    ids = await vectorstore.aadd_texts([dummy_corpus1], [{'source': 'book'}])
    assert ids[0].startswith('test1')

@pytest.mark.asyncio
async def test_aembed_documents_with_similarity_search(vectorstore: RedisVectorStore, chunks: List[str]):
    await vectorstore.aadd_documents(chunks)
    query = "What did King Ulfric Stormborn do in 879"
    results = await vectorstore.asimilarity_search(
        query, 
        k=2, 
        filter=Tag('source') == 'book'
    )
    assert len(results) == 2