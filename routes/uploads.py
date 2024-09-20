import os
import shutil
from typing import List
from fastapi import UploadFile, logger
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from orchestrators.doc.document_ingestor import DocumentIngestor

def format_file_for_storage(uuid: str, conversation_id: str, filename: str):
    return f'files/{uuid}/conversations/{conversation_id}/{filename}'

async def ingest_file(embedding_models: List[BaseEmbedding], uuid: str, conversation_id, upload_file: UploadFile) -> List[str]:
    conversation_id = str(conversation_id)
    path = format_file_for_storage(uuid, conversation_id, upload_file.filename)
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(path, 'wb') as f:
        shutil.copyfileobj(upload_file.file, f)
        
    ingestor = DocumentIngestor(path, embedding_models, uuid, conversation_id)
    embedded_ids = await ingestor.ingest()
    logger.logging.warning(f'EMBEDDED IDs: {len(embedded_ids)}')
    # TODO: make asynchronous
    return embedded_ids