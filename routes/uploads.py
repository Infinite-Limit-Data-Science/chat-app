import os
import asyncio
import shutil
from typing import List, Dict
from fastapi import UploadFile, logger, Request
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from orchestrators.doc.ingestors.ingest import ingest, DocumentIngestor
from routes.configs import get_current_embedding_models

_CURRENT_VECTOR_STORE = 'redis'

def format_file_for_storage(uuid: str, conversation_id: str, filename: str):
    return f'files/{uuid}/conversations/{conversation_id}/{filename}'

async def ingest_file(embedding_models: List[BaseEmbedding], upload_file: UploadFile, data: dict) -> DocumentIngestor:
    path = format_file_for_storage(data['uuid'], str(data['conversation_id']), upload_file.filename)
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(path, 'wb') as f:
        shutil.copyfileobj(upload_file.file, f)

    ingestor = await ingest(
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

    return ingestor

async def ingest_files(request: Request, upload_files: List[UploadFile], data: dict) -> List[DocumentIngestor]:
    embedding_models = await get_current_embedding_models(request)
    tasks = [
        asyncio.create_task(
            ingest_file(embedding_models, upload_file, data)
        ) 
        for upload_file in upload_files
    ]
    ingestors = await asyncio.gather(*tasks)
    return ingestors
