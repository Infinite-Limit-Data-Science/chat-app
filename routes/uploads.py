import os
import asyncio
import shutil
from typing import List, Tuple
from fastapi import UploadFile, Request
from langchain_core.vectorstores import VectorStoreRetriever
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from orchestrators.doc.ingestors.ingest import ingest
from routes.configs import get_current_embedding_models

_CURRENT_VECTOR_STORE = 'redis'

def format_file_for_storage(uuid: str, conversation_id: str, filename: str):
    return f'files/{uuid}/conversations/{conversation_id}/{filename}'

async def ingest_file(embedding_models: List[BaseEmbedding], upload_file: UploadFile, data: dict) -> VectorStoreRetriever:
    """Ingest vectors and return retriever to retrieve them"""
    path = format_file_for_storage(data['uuid'], str(data['conversation_id']), upload_file.filename)
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(path, 'wb') as f:
        shutil.copyfileobj(upload_file.file, f)

    retriever = await ingest(
        path, 
        _CURRENT_VECTOR_STORE, 
        embedding_models, 
        [
            {
                'name': 'uuid',
                'type': 'tag',
                'value': data['uuid'],
            },
            {
                'name': 'conversation_id', 
                'type': 'tag',
                'value': str(data['conversation_id']),
            },
            {
                'name': 'source',
                'type': 'tag',
                'value': upload_file.filename
            },
        ]
    )

    try:
        os.remove(path)
    except OSError as e:
        print(f'Error deleting file {path}: {e}')

    return retriever

async def ingest_files(
        request: Request, 
        upload_files: List[UploadFile], 
        data: dict
    ) -> Tuple[List[VectorStoreRetriever], List[str]]:
    embedding_models = await get_current_embedding_models(request)
    filenames = [upload_file.filename for upload_file in upload_files]
    tasks = [
        asyncio.create_task(
            ingest_file(embedding_models, upload_file, data)
        ) 
        for upload_file in upload_files
    ]
    retrievers = await asyncio.gather(*tasks)
    return retrievers, filenames
