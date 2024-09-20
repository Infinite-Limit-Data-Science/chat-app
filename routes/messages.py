from typing import Dict, Any, Optional, List
from bson import ObjectId
import asyncio
from fastapi import APIRouter, status, Request, Form, Body, Depends, File, UploadFile, logger
from fastapi.responses import StreamingResponse
from models.mongo_schema import ObjectId
from auth.bearer_authentication import get_current_user
from routes.chats import ( 
    get_current_models, 
    get_current_embedding_models, 
    get_prompt_template, 
    chat
)
from orchestrators.chat.llm_models.llm import LLM
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from orchestrators.chat.messages.message_history import MongoMessageHistory
from repositories.base_mongo_repository import base_mongo_factory as factory
from routes.uploads import ingest_file
from models.message import (
    Message,
    MessageSchema,
    BaseMessageSchema,
)

MessageRepo = factory(Message)

router = APIRouter(
    prefix='/conversations', 
    tags=['conversation'],
    dependencies=[Depends(get_current_user)]
)

@router.post(
    '/{conversation_id}/message',
    response_description="Add new message",
    response_model=MessageSchema,
    status_code=status.HTTP_201_CREATED,
    tags=['message']
)
async def create_message(
    request: Request,
    conversation_id: str,
    content: str = Form(...),
    models: LLM = Depends(get_current_models),
    embedding_models: List[BaseEmbedding] = Depends(get_current_embedding_models),
    prompt_template: str = Depends(get_prompt_template),
    upload_file: Optional[UploadFile] = File(None)):
    """Insert new message record in configured database, returning AI Response"""
    conversation_id = ObjectId(conversation_id)
    message_schema = MessageSchema(conversation_id=conversation_id, History=BaseMessageSchema(content=content, type='human'))
    if upload_file:
        await ingest_file(embedding_models, request.state.uuid, conversation_id, upload_file)
    metadata = { 'uuid': request.state.uuid, 'conversation_id': conversation_id }
    run_llm, streaming_handler = await chat(
        prompt_template, 
        models,
        embedding_models,
        metadata,
        message_schema)
    asyncio.create_task(asyncio.to_thread(run_llm))
    return StreamingResponse(streaming_handler.get_streamed_response(), media_type="text/plain")

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
        message := await MessageRepo.find(request.state.uuid, options={'conversation_id': conversation_id, '_id': ObjectId(id)})
    ) is not None:
        return message
    return {'error': f'Message {id} not found'}, 404

# Note Conversational AI does not allow the edit of existing messages
@router.delete(
    '/{conversation_id}/message/{id}', 
    response_description='Delete a message',
    tags=['message']
)
async def delete_message(request: Request, conversation_id: str, id: str):
    """Remove a single message record from the database."""
    if (
        deleted_message := await MessageRepo.delete(id, options={'conversation_id': ObjectId(conversation_id)})
    ) is not None:
        return deleted_message  
    return { 'error': f'Conversation {id} not found'}, 404