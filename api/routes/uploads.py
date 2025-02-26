import os
import time
from typing import List, Tuple, Dict
from fastapi import UploadFile
from langchain_core.vectorstores import VectorStoreRetriever
from ..langchain_doc import ingest, BaseEmbedding
from ..logger import logger
from .configs import ChatBotConfig

async def ingest_files(
    upload_files: List[UploadFile], 
    chat_bot_config: ChatBotConfig,
) -> List[str]:
    if not (vector_store := os.getenv('VECTOR_STORE')):
        raise ValueError('Expected `VECTOR_STORE` to be defined')
    
    metadata = {
        'uuid': chat_bot_config.user_config.uuid,
        'conversation_id': str(chat_bot_config.user_config.session_id)
    }
    start_time = time.time()
    filenames = await ingest(
        vector_store, 
        upload_files, 
        chat_bot_config.embeddings,
        metadata,
    )
    duration = time.time() - start_time
    logger.info(f'Ingestion time for {filenames}: {duration:.2f} seconds')

    return filenames