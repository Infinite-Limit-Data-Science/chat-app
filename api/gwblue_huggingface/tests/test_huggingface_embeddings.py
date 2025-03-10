import pytest
import os
import json
from typing import List, Iterator
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_redis import RedisConfig
from langchain_redis import RedisVectorStore
from redisvl.query.filter import Tag
from ..huggingface_embeddings import HuggingFaceEmbeddings
from ..huggingface_transformer_tokenizers import (
    BgeLargePretrainedTokenizer,
    VLM2VecFullPretrainedTokenizer
) 
from ..huggingface_inference_server_config import HuggingFaceTEIConfig
from .corpus import dummy_corpus1

load_dotenv()

def _model_config(model_type: str, model_name: str) -> str:
    models = json.loads(os.environ[model_type])
    model = next((model for model in models if model["name"] == model_name), None)
    if not model:
        raise ValueError(f"Model {model_name} does not exist in {model_type}")

    return {
        'name': model['name'],
        'url': model['endpoints'][0]['url']
    }

@pytest.fixture
def tei_self_hosted_config() -> HuggingFaceTEIConfig:
    config = _model_config("EMBEDDING_MODELS", "BAAI/bge-large-en-v1.5")

    return HuggingFaceTEIConfig(
        name=config['name'],
        url=config['url'],
        auth_token=os.environ['TEST_AUTH_TOKEN'],        
        max_batch_tokens=32768,
        max_client_batch_size=128,
        max_batch_requests=64,
        auto_truncate=True
    )

@pytest.fixture
def tei_self_hosted_config_vision() -> HuggingFaceTEIConfig:
    config = _model_config("EMBEDDING_MODELS", "TIGER-Lab/VLM2Vec-Full")
    
    return HuggingFaceTEIConfig(
        name=config['name'],
        url=config['url'],
        auth_token=os.environ['TEST_AUTH_TOKEN'],
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
def chunks_multimodal(tei_self_hosted_config: HuggingFaceTEIConfig) -> List[str]:
    # pdf with images: START HERE
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