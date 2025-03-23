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
from ...gwblue_chat_bot.chat_bot_config import (
    LLMConfig,
    EmbeddingsConfig,
    RedisVectorStoreConfig,
    MongoMessageHistoryConfig,
    ChatBotConfig,
)

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
def chat_bot_config(redis_client: Redis) -> ChatBotConfig:
    config = _model_config("MODELS", "meta-llama/Llama-3.2-11B-Vision-Instruct")
    llm_config = LLMConfig(
        **{
            "model": config["name"],
            "endpoint": config["url"],
            "token": os.environ["TEST_AUTH_TOKEN"],
            "parameters": {"temperature": 0.8},
            "provider": config["provider"],
        }
    )

    config = _model_config("MODELS", "meta-llama/Llama-Guard-3-8B")
    guardrails_config = LLMConfig(
        **{
            "model": config["name"],
            "endpoint": config["url"],
            "token": os.environ["TEST_AUTH_TOKEN"],
            "parameters": {"temperature": 0},
            "provider": config["provider"],
        }
    )

    config = _model_config("EMBEDDING_MODELS", "TIGER-Lab/VLM2Vec-Full")
    embeddings_config = EmbeddingsConfig(
        **{
            "model": config["name"],
            "endpoint": config["url"],
            "token": os.environ["TEST_AUTH_TOKEN"],
            "max_batch_tokens": 32768,
            "provider": config["provider"],
        }
    )

    metadata_schema = json.loads(os.environ["VECTOR_STORE_SCHEMA"])
    vectorstore_config = RedisVectorStoreConfig(
        **{
            "client": redis_client,
            "metadata_schema": metadata_schema,
        }
    )

    message_history_config = MongoMessageHistoryConfig(
        name=os.environ["DATABASE_NAME"],
        url=os.environ["MONGODB_URL"],
        collection_name="messages",
        session_id_key="conversation_id",
    )

    return ChatBotConfig(
        llm=llm_config,
        embeddings=embeddings_config,
        guardrails=guardrails_config,
        vectorstore=vectorstore_config,
        message_history=message_history_config,
    )

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
        "uuid": str(ObjectId()),
        "conversation_id": str(uuid4()),
    }

@pytest.mark.asyncio
async def test_pdf_embed(
    embeddings: HuggingFaceEmbeddings,
    chat_bot_config: ChatBotConfig,
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
        embeddings_config=chat_bot_config.embeddings,
    )

    start_time = time.perf_counter()
    ids = await ingestor.ingest()
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    print(f"Ingestion took {elapsed:.2f} seconds")

    assert len(ids) > 0
