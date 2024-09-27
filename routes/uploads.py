import os
import shutil
from typing import List
from fastapi import UploadFile, logger
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from orchestrators.doc.ingestors.ingest import ingest

_CURRENT_VECTOR_STORE = 'redis'

def format_file_for_storage(uuid: str, conversation_id: str, filename: str):
    return f'files/{uuid}/conversations/{conversation_id}/{filename}'

async def ingest_file(embedding_models: List[BaseEmbedding], upload_file: UploadFile, data: dict) -> List[str]:
    path = format_file_for_storage(data['uuid'], str(data['conversation_id']), upload_file.filename)
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(path, 'wb') as f:
        shutil.copyfileobj(upload_file.file, f)
    embedded_ids = await ingest(path, _CURRENT_VECTOR_STORE, embedding_models, {
            'metadata': {
                'uuid': data['uuid'], 
                'conversation_id': data['conversation_id'],
                'schema': [
                    {
                        'name': 'uuid', 
                        'type': 'tag'
                    },
                    {
                        'name': 'conversation_id', 
                        'type': 'tag'
                    },
                ]
            },
            'configurable': {
                'session_id': data['conversation_id'],
            }
    })
    logger.logging.warning(f'EMBEDDED IDs: {len(embedded_ids)}')
    return embedded_ids