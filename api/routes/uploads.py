import os
import time
import shutil
import logging
from pathlib import Path
import asyncio
from functools import partial
from fastapi import UploadFile
from typing import Tuple, List, Dict, Any, BinaryIO, Protocol, Annotated
from typing_extensions import Doc
from ..logger import logger
from .configs import ChatBotConfig
from ..gwblue_huggingface import HuggingFaceEmbeddings
from ..gwblue_chat_bot.chat_bot_config import EmbeddingsConfig, RedisVectorStoreConfig
from ..gwblue_ingestion_pipeline import (
    LazyPdfIngestor,
    LazyWordIngestor,
    LazyPowerPointIngestor,
    LazyTextIngestor,
)

INGESTION_FACTORIES = {
    "pdf": LazyPdfIngestor,
    "docx": LazyWordIngestor,
    "pptx": LazyPowerPointIngestor,
    "txt": LazyTextIngestor,
}


class FileLike(Protocol):
    @property
    def filename(self) -> Annotated[str, Doc("Name of binary object")]: ...

    @property
    def file(self) -> Annotated[BinaryIO, Doc("Binary object")]: ...


def load_embeddings(embeddings_config: EmbeddingsConfig):
    return HuggingFaceEmbeddings(
        base_url=embeddings_config.endpoint,
        credentials=embeddings_config.token,
        provider=embeddings_config.provider,
        model=embeddings_config.model,
    )


def generate_path(fields: Dict[str, str], filename: str) -> Path:
    path_components = [f"{key}/{value}" for key, value in fields.items()]
    return Path("files").joinpath(*path_components, filename)


async def ingest(
    files: List[FileLike],
    *,
    embeddings_model_config: EmbeddingsConfig,
    vector_store_config: RedisVectorStoreConfig,
    metadata: List[dict],
) -> List[Dict[str, Any]]:
    embeddings = load_embeddings(embeddings_model_config)

    filenames = []
    paths = []
    ingestors = []
    metadatas = []

    try:
        for file in files:
            path = generate_path(metadata, file.filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                shutil.copyfileobj(file.file, f)
            paths.append(path)
            filenames.append(file.filename)

            metadata = {
                **metadata,
                "conversation_id": str(metadata["conversation_id"]),
                "source": file.filename,
            }
            metadatas.append(metadata)
            ingestor = partial(
                INGESTION_FACTORIES[os.path.splitext(file.filename)[1][1:]],
                path,
                embeddings=embeddings,
                metadata=metadata,
                vector_config=vector_store_config,
                embeddings_config=embeddings_model_config,
            )
            ingestors.append(ingestor)

        tasks = [asyncio.create_task(ingestor().ingest()) for ingestor in ingestors]
        ids: List[List[str]] = await asyncio.gather(*tasks)

        if not len(ids) == len(files):
            raise AssertionError(f"Expected to ingest {len(files)} files with {files}")

        return metadatas
    finally:
        for path in paths:
            try:
                os.remove(path)
            except OSError as e:
                logging.warning(f"Error deleting file {path}: {e}")


async def ingest_files(
    *,
    files: List[UploadFile],
    config: ChatBotConfig,
    metadata: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    start_time = time.time()
    metadatas = await ingest(
        files,
        embeddings_model_config=config.embeddings,
        vector_store_config=config.vectorstore,
        metadata=metadata,
    )
    duration = time.time() - start_time

    filenames = [metadata["source"] for metadata in metadatas]
    logger.info(f"Ingestion time for {filenames}: {duration:.2f} seconds")

    return metadatas, filenames
