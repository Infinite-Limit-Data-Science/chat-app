import os
import shutil
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, status, Request, Body, Depends, File, UploadFile, logger
from fastapi.responses import StreamingResponse
from auth.bearer_authentication import get_current_user
from models.conversation import Conversation
from models.setting import (
    Setting,
    SettingSchema
)
from models.message import (
    MessageSchema,
    UpdateMessageSchema,
)
from models.model_config import (
    ModelConfigSchema,
    ModelConfig,
)
from repositories.message_mongo_repository import MessageMongoRepository as MessageRepo
from repositories.base_mongo_repository import base_mongo_factory as factory
from orchestrators.chat.llm_chat import LLMChat
from orchestrators.doc.document_ingestor import DocumentIngestor
from orchestrators.chat.llm import LLM
from orchestrators.chat.factories import FACTORIES

ConversationRepo = factory(Conversation)
SettingRepo = factory(Setting)
ModelConfigRepo = factory(ModelConfig)

router = APIRouter(
    prefix='/conversations', 
    tags=['conversation'],
    dependencies=[Depends(get_current_user)]
)

async def get_current_models(request: Request) -> List[LLM]:
    setting_schema = SettingSchema(SettingRepo.find(options={request.state.uuid_name: request.state.uuid}))
    model_config_schema = ModelConfigSchema(await ModelConfigRepo.find(options={'activeModel': setting_schema.activeModel}))    
    models = [
        FACTORIES[endpoint.type](**{
            'name': model_config_schema.name,
            'description': model_config_schema.description,
            'default_prompt': model_config_schema.default_prompt,
            'parameters': model_config_schema.parameters,
            'endpoint': endpoint
        })
        for endpoint in model_config_schema.endpoints
    ]
    return models


@router.post(
    '/{conversation_id}/message',
    response_description="Add new message",
    response_model=MessageSchema,
    status_code=status.HTTP_201_CREATED,
    tags=['message']
)
async def create_message(request: Request, conversation_id: str, message_schema: MessageSchema = Body(...), models: LLM = Depends(get_current_models)):
    """Insert new message record in configured database, returning resource created"""
    if (
        _ := await ConversationRepo.update(id, schema=message_schema)
    ) is not None:
        llm_chat = LLMChat(models)
        # START HERE
        docs = llm_chat.template_method(message_schema.content)
        await MessageRepo.create(conversation_id, MessageSchema(content=str(chat_bot), modelDetail=user_message.modelDetail))
        if "application/json" in request.headers.get("Accept"):
            return { 'docs': [doc.page_content for doc in docs]}
        if 'text/event-stream' in request.headers.get("Accept"):
            async def stream_response():
                for doc in docs:
                    yield doc.page_content.encode("utf-8")
            return StreamingResponse(stream_response())
    return {'error': f'Conversation not updated'}, 400

@router.get(
    '/{conversation_id}/message/{id}',
    response_description="Get a single message",
    response_model=MessageSchema,
    response_model_by_alias=False,
    tags=['message']
)
async def get_message(request: Request, conversation_id: str, id: str):
    """Get message record from configured database by id"""
    if (
        message := await MessageRepo.find(request.state.uuid, conversation_id, id)
    ) is not None:
        return message
    return {'error': f'Message {id} not found'}, 404

@router.put(
    '/{conversation_id}/message/{id}',
    response_description="Update a single message",
    response_model=MessageSchema,
    response_model_by_alias=False,
    tags=['message']
)
async def update_message(request: Request, conversation_id: str, id: str, message: UpdateMessageSchema = Body(...)):
    """Update individual fields of an existing message record and return modified fields to client."""
    if (
        updated_message := await MessageRepo.update(request.state.uuid, conversation_id, id, message)
    ) is not None:
        return updated_message
    return {'error': f'Conversation {id} not found'}, 404
    
@router.delete(
    '/{conversation_id}/message/{id}', 
    response_description='Delete a message',
    tags=['message']
)
async def delete_message(request: Request, conversation_id: str, id: str):
    """Remove a single message record from the database."""
    if (
        deleted_message := await MessageRepo.delete(request.state.uuid, conversation_id, id)
    ) is not None:
        return deleted_message  
    return { 'error': f'Conversation {id} not found'}, 404

def format_file_for_storage(uuid: str, conversation_id: str, id: str, filename: str):
    return f'files/{uuid}/conversations/{conversation_id}/messages/{id}/{filename}'

async def upload_file(request: Request, conversation_id: str, id: str, upload_file: UploadFile = File(...)) -> Optional[Dict[str,Any]]:
    uuid = request.state.uuid
    if (
        message_dict := await MessageRepo.find(uuid, conversation_id, id)
    ) is not None:
        path = format_file_for_storage(uuid, conversation_id, id, upload_file.filename)
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(path, 'wb') as f:
            shutil.copyfileobj(upload_file.file, f)
        merged_files = ([] if message_dict['files'] is None else message_dict['files']) + [path]
        return await MessageRepo.update(uuid, conversation_id, id, UpdateMessageSchema(files=merged_files))

@router.post(
    '/{conversation_id}/message/{id}/upload',
    response_model=MessageSchema,
    response_description='Upload a file',
    tags=['message']
)
async def upload(request: Request, conversation_id: str, id: str, upload_file: UploadFile = File(...), message_dict: Optional[Dict[str,Any]] = Depends(upload_file)):
    if not message_dict:
        return {'error': f'Message {id} not found'}, 404
    file = format_file_for_storage(request.state.uuid, conversation_id, id, upload_file.filename)
    ingestor = DocumentIngestor(file, request.state.uuid, conversation_id, id)
    # TODO: make asynchronous
    embedded_ids = ingestor.ingest()
    logger.logging.warning(f'EMBEDDED IDs: {len(embedded_ids)}')
    return message_dict