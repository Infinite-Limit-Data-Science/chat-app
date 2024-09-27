import logging
from typing import Dict, Any, Optional, List
from bson import ObjectId
from fastapi import (
    APIRouter, 
    status, 
    Request, 
    Form, 
    Body, 
    Depends, 
    File, 
    UploadFile, 
    logger,
)
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
from repositories.base_mongo_repository import base_mongo_factory as factory
from routes.uploads import ingest_file
from models.message import (
    Message,
    MessageSchema,
)
from repositories.conversation_mongo_repository import ConversationMongoRepository as ConversationRepo

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
    data = { 'uuid': request.state.uuid, 'conversation_id': conversation_id }
    if upload_file:
        await ingest_file(embedding_models, upload_file, data)
    message_schema = MessageSchema(type='human', content=content, conversation_id=conversation_id)
    llm_stream = await chat(
        prompt_template, 
        models,
        embedding_models,
        data,
        message_schema)
    return StreamingResponse(llm_stream(), media_type="text/plain", headers={"X-Accel-Buffering": "no"})

@router.get(
    '/{conversation_id}/message/{id}',
    response_description="Get a single message",
    response_model=MessageSchema,
    response_model_by_alias=False,
    tags=['message']
)
async def get_message(request: Request, conversation_id: str, id: str):
    """Get message record from configured database by id"""
    logging.warning(f'message id {id} AND conversation id {conversation_id}')
    if (
        message := await MessageRepo.find_one(id, options={'conversation_id': ObjectId(conversation_id) })
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
        response = await ConversationRepo.remove_from_field(conversation_id, options={'message_ids': ObjectId(id) })
        return deleted_message
    return { 'error': f'Conversation {id} not found'}, 404