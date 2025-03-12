import pytest
import os
import json
from pathlib import Path
from typing import List, Iterator
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_redis import RedisConfig
from redisvl.query.filter import Tag
from ..huggingface_embeddings import HuggingFaceEmbeddings
from ..huggingface_transformer_tokenizers import (
    BgeLargePretrainedTokenizer,
    VLM2VecFullPretrainedTokenizer
) 
from langchain_community.document_loaders import PyPDFLoader
from ...gwblue_document_loaders.parsers import Base64BlobParser
from ...gwblue_text_splitters import MixedContentTextSplitter
from ...gwblue_vectorstores.redis.multimodal_vectorstore import (
    MultiModalVectorStore
)
from .corpus import dummy_corpus1

load_dotenv()

def _model_config(model_type: str, model_name: str) -> str:
    models = json.loads(os.environ[model_type])
    model = next((model for model in models if model["name"] == model_name), None)
    if not model:
        raise ValueError(f"Model {model_name} does not exist in {model_type}")

    return {
        'name': model['name'],
        'url': model['endpoints'][0]['url'],
        'provider': model['endpoints'][0]['provider'],
    }

@pytest.fixture
def text_embeddings() -> HuggingFaceEmbeddings:
    config = _model_config("EMBEDDING_MODELS", "BAAI/bge-large-en-v1.5")

    return HuggingFaceEmbeddings(
        base_url=config['url'],
        credentials=os.environ['TEST_AUTH_TOKEN'],
        provider=config['provider'],
        model=config['name'],
    )

@pytest.fixture
def embeddings() -> HuggingFaceEmbeddings:
    config = _model_config("EMBEDDING_MODELS", "TIGER-Lab/VLM2Vec-Full")

    return HuggingFaceEmbeddings(
        base_url=config['url'],
        credentials=os.environ['TEST_AUTH_TOKEN'],
        provider=config['provider'],
        model=config['name'],
    )

@pytest.fixture
def bge_tokenizer() -> BgeLargePretrainedTokenizer:
    return BgeLargePretrainedTokenizer()

@pytest.fixture
def vlm_tokenizer() -> VLM2VecFullPretrainedTokenizer:
    return VLM2VecFullPretrainedTokenizer()

@pytest.fixture
def text_chunks(vlm_tokenizer: VLM2VecFullPretrainedTokenizer) -> List[str]:
    docs = [Document(page_content=dummy_corpus1, metadata={'source': 'book', 'page': 0})]
    sequence_length = 2000
    overlap = int(sequence_length * 0.05)
    len_function = lambda text: len(vlm_tokenizer.tokenizer.encode(text))

    mixed_content_splitter = MixedContentTextSplitter(
        chunk_size=sequence_length,
        chunk_overlap=overlap,
        length_function=len_function,
        metadata={'uuid': '1', 'conversation_id': '1'},
    )
    chunks = mixed_content_splitter.split_documents(docs)
    
    return chunks

@pytest.fixture
def mixed_message_chunks(vlm_tokenizer: VLM2VecFullPretrainedTokenizer) -> List[str]:
    pdf_path = Path(__file__).parent / 'assets' / 'jpeg.pdf'
    loader = PyPDFLoader(
        pdf_path,
        extract_images=True,
        images_parser=Base64BlobParser(),
        images_inner_format="raw",
        mode="page",
    )
    docs = loader.load()

    sequence_length = 2000
    overlap = int(sequence_length * 0.05)
    len_function = lambda text: len(vlm_tokenizer.tokenizer.encode(text))

    mixed_content_splitter = MixedContentTextSplitter(
        chunk_size=sequence_length,
        chunk_overlap=overlap,
        length_function=len_function,
        metadata={'uuid': '1', 'conversation_id': '1'},
    )
    chunks = mixed_content_splitter.split_documents(docs)
    
    return chunks    


@pytest.fixture
def vectorstore(
    embeddings: HuggingFaceEmbeddings, 
    vlm_tokenizer: VLM2VecFullPretrainedTokenizer
) -> Iterator[MultiModalVectorStore]:
    config = RedisConfig(
        index_name="test1",
        redis_url=os.environ['REDIS_URL'],
        metadata_schema=[
            {"name": "source", "type": "tag"},
        ],
        embedding_dimensions=vlm_tokenizer.dimensions
    )

    store = MultiModalVectorStore(embeddings, config=config)

    yield store

    store.index.clear()
    store.index.delete(drop=True)

# 
# @pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_bge_embed_documents(
    text_embeddings: text_embeddings,
    bge_tokenizer: BgeLargePretrainedTokenizer,
):
    embedded_vectors = text_embeddings.embed_documents([dummy_corpus1])
    assert len(embedded_vectors[0]) == bge_tokenizer.dimensions

def test_vlm_embed_documents(
    embeddings: HuggingFaceEmbeddings,
    vlm_tokenizer: VLM2VecFullPretrainedTokenizer,
):
    embedded_vectors = embeddings.embed_documents([dummy_corpus1])
    assert len(embedded_vectors[0]) == vlm_tokenizer.dimensions

def test_embed_multiple_documents(embeddings: HuggingFaceEmbeddings):
    embedded_vectors = embeddings.embed_documents([dummy_corpus1, dummy_corpus1.upper()])
    assert len(embedded_vectors) > 0

def test_embed_query(embeddings: HuggingFaceEmbeddings):
    embedded_vectors = embeddings.embed_query(dummy_corpus1)
    assert len(embedded_vectors) > 0

def test_embed_documents_in_vector_db(vectorstore: MultiModalVectorStore):
    ids = vectorstore.add_texts([dummy_corpus1], [{'source': 'book'}])
    assert ids[0].startswith('test1')

def test_embed_documents_with_similarity_search(vectorstore: MultiModalVectorStore, text_chunks: List[str]):
    vectorstore.add_documents(text_chunks)
    query = "What did King Ulfric Stormborn do in 879"
    results = vectorstore.similarity_search(
        query, 
        k=2, 
        filter=Tag('source') == 'book'
    )

    assert len(results) == 2

def test_embed_documents_with_similarity_search_with_score(vectorstore: MultiModalVectorStore, text_chunks: List[str]): 
    vectorstore.add_documents(text_chunks)
    query = "What did King Ulfric Stormborn do in 879"    
    results = vectorstore.similarity_search_with_score(
        query, 
        k=2, 
        filter=Tag('source') == 'book'
    )

    score1 = results[0][1]
    score2 = results[1][1]

    assert score1 < score2

def test_embed_documents_with_max_marginal_relevance_search(
    vectorstore: MultiModalVectorStore, 
    text_chunks: List[str]
):
    vectorstore.add_documents(text_chunks)
    query = "What did King Ulfric Stormborn do in 879"    
    results = vectorstore.max_marginal_relevance_search(
        query, 
        k=2, 
        filter=Tag('source') == 'book'
    )

    assert len(results) == 2

def test_embed_documents_with_similarity_search_by_vector(
    embeddings: HuggingFaceEmbeddings, 
    vectorstore: MultiModalVectorStore, 
    text_chunks: List[str]
):
    vectorstore.add_documents(text_chunks)
    
    query = "What did King Ulfric Stormborn do in 879"
    float_32_1024_dimensional_bge_vector = embeddings.embed_documents([query])[0]
    results = vectorstore.similarity_search_by_vector(
        float_32_1024_dimensional_bge_vector, 
        k=2, 
        filter=Tag('source') == 'book'
    )
    assert len(results) == 2

def test_embed_documents_with_similarity_search_with_score_by_vector(
    embeddings: HuggingFaceEmbeddings, 
    vectorstore: MultiModalVectorStore, 
    text_chunks: List[str]
):
    vectorstore.add_documents(text_chunks)

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

def test_embed_documents_with_max_marginal_relevance_search_by_vector(
    embeddings: HuggingFaceEmbeddings, 
    vectorstore: MultiModalVectorStore, 
    text_chunks: List[str]
):
    vectorstore.add_documents(text_chunks)

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
async def test_aembed_documents_in_vector_db(vectorstore: MultiModalVectorStore):
    ids = await vectorstore.aadd_texts([dummy_corpus1], [{'source': 'book'}])
    assert ids[0].startswith('test1')

@pytest.mark.asyncio
async def test_aembed_documents_with_similarity_search(
    vectorstore: MultiModalVectorStore, 
    text_chunks: List[str]
):
    await vectorstore.aadd_documents(text_chunks)
    query = "What did King Ulfric Stormborn do in 879"
    results = await vectorstore.asimilarity_search(
        query, 
        k=2, 
        filter=Tag('source') == 'book'
    )
    assert len(results) == 2

def test_embed_multimodal_documents_with_similarity_search(
    vectorstore: MultiModalVectorStore, 
    mixed_message_chunks: List[str]
):
    vectorstore.add_documents(mixed_message_chunks)
    query = "Describe the document"
    results = vectorstore.similarity_search(
        query, 
        k=2, 
        filter=Tag('source') == "jpeg.pdf"
    )

    assert len(results) == 2

def test_embed_multimodal_documents_with_max_marginal_relevance_search(
    vectorstore: MultiModalVectorStore, 
    mixed_message_chunks: List[str]
):
    vectorstore.add_documents(mixed_message_chunks)
    query = "Describe the document"
    results = vectorstore.max_marginal_relevance_search(
        query, 
        k=2, 
        filter=Tag('source') == 'jpeg.pdf'
    )

    assert len(results) == 2

@pytest.mark.asyncio
async def test_aembed_multimodal_documents_with_similarity_search(
    vectorstore: MultiModalVectorStore, 
    mixed_message_chunks: List[str]
):
    await vectorstore.aadd_documents(mixed_message_chunks)
    query = "Describe the document"
    results = await vectorstore.asimilarity_search(
        query, 
        k=2, 
        filter=Tag('source') == "jpeg.pdf"
    )

    assert len(results) == 2