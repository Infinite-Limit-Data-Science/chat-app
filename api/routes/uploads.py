import os
import time
from typing import Tuple, List, Dict, Any
from fastapi import UploadFile
from langchain_core.vectorstores import VectorStoreRetriever
from ..langchain_doc import ingest, BaseEmbedding
from ..logger import logger
from .configs import ChatBotConfig

async def ingest_files(
    *,
    files: List[UploadFile], 
    config: ChatBotConfig,
    metadata: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not (vector_store := os.getenv('VECTOR_STORE')):
        raise ValueError('Expected `VECTOR_STORE` to be defined')
    
    start_time = time.time()
    metadatas = await ingest(
        vector_store, 
        files, 
        config.embeddings,
        config.vectorstore,
        metadata,
    )
    duration = time.time() - start_time
    filenames = [metadata['source'] for metadata in metadatas]
    logger.info(f'Ingestion time for {filenames}: {duration:.2f} seconds')

    return metadatas, filenames
