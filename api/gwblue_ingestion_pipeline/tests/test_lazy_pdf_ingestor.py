from typing import Dict, Any
import os
import json
import time
import pytest
from pathlib import Path
from uuid import uuid4
from bson import ObjectId
from redis.client import Redis
from redis.connection import ConnectionPool
from ..lazy_pdf_ingestor import LazyPdfIngestor
from ...gwblue_huggingface import HuggingFaceEmbeddings

def _model_config(model_type: str, model_name: str) -> str:
    models = json.loads(os.environ[model_type])
    model = next((model for model in models if model["name"] == model_name), None)
    if not model:
        raise ValueError(f"Model {model_name} does not exist in {model_type}")

    return {
        "name": model["name"],
        "url": model["endpoints"][0]["url"],
        "provider": model["endpoints"][0]["provider"],
    }

@pytest.fixture
def redis_client() -> Redis:
    redis_client = Redis.from_pool(
        ConnectionPool.from_url(
            os.environ["REDIS_URL"], max_connections=50, socket_timeout=30.0
        )
    )
    return redis_client

@pytest.fixture
def embeddings() -> HuggingFaceEmbeddings:
    config = _model_config("EMBEDDING_MODELS", "TIGER-Lab/VLM2Vec-Full")

    return HuggingFaceEmbeddings(
        base_url=config["url"],
        credentials=os.environ["TEST_AUTH_TOKEN"],
        provider=config["provider"],
        model=config["name"],
    )

assets_dir = Path(__file__).parent / "assets"

@pytest.fixture
def calculus_book1_path() -> Path:
    return assets_dir / "CalculusBook1.pdf"

@pytest.fixture
def message_metadata() -> Dict[str, Any]:
    return {
        "uuid": str(uuid4()),
        "conversation_id": str(uuid4()),
    }

@pytest.mark.asyncio
async def test_calculus_book_embed(
    embeddings: HuggingFaceEmbeddings,
    message_metadata: Dict[str, Any],
    calculus_book1_path: Path,
):
    metadata = {
        **message_metadata,
        "source": "CalculusBook1.pdf",
    }

    ingestor = LazyPdfIngestor(
        calculus_book1_path,
        embeddings=embeddings,
        metadata=metadata,
        vector_config=chat_bot_config.vectorstore,
    )

    start_time = time.perf_counter()
    ids = await ingestor.ingest()
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    print(f"Ingestion took {elapsed:.2f} seconds")

    assert len(ids) > 0

@pytest.mark.asyncio
async def test_calculus_book_inheritable_embed(
    embeddings: HuggingFaceEmbeddings,
    message_metadata: Dict[str, Any],
    calculus_book1_path: Path,
):
    ...