from typing import (
    Dict,
    Any,
)
import pytest
import uuid
import time
import json
import os
from pathlib import Path
from redis.client import Redis
from redis.connection import ConnectionPool
from langchain_redis import RedisConfig
from ...gwblue_text_splitters.mixed_content_text_splitter import MixedContentTextSplitter
from ...gwblue_huggingface.huggingface_transformer_tokenizers import (
    get_tokenizer_by_prefix,
    BaseLocalTokenizer,
)
from ...gwblue_document_loaders.loaders.extended_pypdf_loader import ExtendedPyPDFLoader
from ...gwblue_document_loaders.parsers.base64_blob_parser import Base64BlobParser
from ...gwblue_vectorstores.redis import MultiModalVectorStore
from ...gwblue_vectorstores.redis.config import VectorStoreSchema
from ...gwblue_vectorstores.redis.docstore import RedisDocStore
from ...gwblue_huggingface import HuggingFaceEmbeddings
from ..streaming_parent_document_retriever import StreamingParentDocumentRetriever

def _model_config(model_type: str, model_name: str) -> Dict[str, str]:
    models = json.loads(os.environ[model_type])
    model = next((model for model in models if model["name"] == model_name), None)
    if not model:
        raise ValueError(f"Model {model_name} does not exist in {model_type}")

    return {
        "name": model["name"],
        "url": model["endpoints"][0]["url"],
        "provider": model["endpoints"][0]["provider"],
    }

assets_dir = Path(__file__).parent / "assets"

@pytest.fixture
def calculus_book1_path() -> Path:
    return assets_dir / "CalculusBook1.pdf"

@pytest.fixture
def redis_client() -> Redis:
    redis_client = Redis.from_pool(
        ConnectionPool.from_url(
            os.environ["REDIS_URL"], max_connections=50, socket_timeout=30.0
        )
    )
    return redis_client

@pytest.fixture
def vlm_tokenizer() -> BaseLocalTokenizer:
    return get_tokenizer_by_prefix("TIGER-Lab/VLM2Vec-Full")

@pytest.fixture
def embeddings() -> HuggingFaceEmbeddings:
    config = _model_config("EMBEDDING_MODELS", "TIGER-Lab/VLM2Vec-Full")

    return HuggingFaceEmbeddings(
        base_url=config["url"],
        credentials=os.environ["TEST_AUTH_TOKEN"],
        provider=config["provider"],
        model=config["name"],
    )

@pytest.fixture
def message_metadata() -> Dict[str, Any]:
    return {
        "uuid": str(uuid.uuid4()),
        "conversation_id": str(uuid.uuid4()),
    }

@pytest.mark.asyncio
async def test_calculus_book_split_by_500(
    embeddings: HuggingFaceEmbeddings,
    redis_client: Redis,
    message_metadata: Dict[str, Any],
    calculus_book1_path: Path,
    vlm_tokenizer: BaseLocalTokenizer,
):
    metadata = {
        **message_metadata,
        "source": "CalculusBook1.pdf",
    }

    loader = ExtendedPyPDFLoader(
        calculus_book1_path,
        extract_images=True,
        images_parser=Base64BlobParser(),
        images_inner_format="raw",
        mode="page",
    )

    docs = loader.lazy_load()

    config = RedisConfig(
        **{
            "redis_client": redis_client,
            "metadata_schema": json.loads(os.environ['VECTOR_STORE_SCHEMA']), # now includes doc_id
            "embedding_dimensions": vlm_tokenizer.vector_dimension_length,
            **VectorStoreSchema().model_dump(),
        }
    )

    vectorstore = MultiModalVectorStore(embeddings, config=config)
    docstore = RedisDocStore(redis_client)

    parent_text_splitter = MixedContentTextSplitter(
        vlm_tokenizer.tokenizer,
        chunk_size=2000,
        metadata=metadata,
    )

    child_text_splitter = MixedContentTextSplitter(
        vlm_tokenizer.tokenizer,
        chunk_size=500,
        metadata=metadata,
    )

    documents = parent_text_splitter.split_documents(docs)

    retriever = StreamingParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_text_splitter,   
    )

    requests = vlm_tokenizer.max_batch_tokens_forward_pass // vlm_tokenizer.sequence_length_forward_pass

    start_time = time.time()
    ids = await retriever.aadd_document_batch(
        documents=documents,
        max_requests=requests
    )
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Number of documents: {len(ids)}")
    print(f"Time elapsed: {elapsed:.2f} seconds")

    assert len(ids) > 0
