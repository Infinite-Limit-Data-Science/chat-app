import logging
import os
import time
from typing import List, Tuple
from fastapi import UploadFile
from langchain_core.vectorstores import VectorStoreRetriever
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from orchestrators.doc.ingestors.ingest import ingest

async def ingest_files(
    embedding_models: List[BaseEmbedding], 
    upload_files: List[UploadFile], 
    data: dict,
) -> Tuple[List[VectorStoreRetriever], List[str]]:
    if not (vector_store := os.getenv('VECTOR_STORE')):
        raise ValueError('Expected `REDIS_STORE` to be defined')
    
    data = {
        **data,
        'conversation_id': str(data['conversation_id']),
    }
    start_time = time.time()
    retrievers, filenames = await ingest(vector_store, upload_files, embedding_models, data)
    duration = time.time() - start_time
    logging.warning(f'Ingestion time for {filenames}: {duration:.2f} seconds')

    return retrievers, filenames