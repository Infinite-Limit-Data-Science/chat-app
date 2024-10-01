import logging
from typing import Annotated, List, Optional, Union
from fastapi import ( 
    APIRouter, 
    status, 
    Request, 
    Query, 
    Body, 
    Form, 
    Depends, 
    File, 
    UploadFile, 
    logger
)
from auth.bearer_authentication import get_current_user
from routes.chats import chat 
from routes.configs import (
    get_current_models, 
    get_current_embedding_models, 
    get_prompt_template, 
    
)
from routes.uploads import ingest_file
from orchestrators.chat.llm_models.llm import LLM
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from repositories.conversation_mongo_repository import ConversationMongoRepository as ConversationRepo
from models.conversation import (
    ConversationSchema,
    CreateConversationSchema,
    ConversationCollectionSchema, 
    UpdateConversationSchema,
)
from models.message import MessageSchema
from fastapi.responses import StreamingResponse

router = APIRouter(
    prefix='/conversations', 
    tags=['conversation'],
    dependencies=[Depends(get_current_user)]
)
@router.get(
    '/',
    response_description='List all conversations',
    response_model=ConversationCollectionSchema,
    response_model_by_alias=False,
)
async def conversations(request: Request, record_offset: int = Query(0, description='record offset', alias='offset'), record_limit: int = Query(20, description="record limit", alias='limit')):
    """List conversations by an offset and limit"""
    conversations = await ConversationRepo.all(options={request.state.uuid_name: request.state.uuid}, offset=record_offset, limit=record_limit)
    return ConversationCollectionSchema(conversations=conversations)

@router.post(
    '/',
    response_description="Add new conversation",
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    request: Request,
    content: str = Form(...),
    # conversation: Annotated[ConversationSchema, Form()],
    # message: Annotated[MessageSchema, Form()],
    models: List[LLM] = Depends(get_current_models),
    embedding_models: List[BaseEmbedding]  = Depends(get_current_embedding_models),
    prompt_template: str = Depends(get_prompt_template),
    upload_file: Optional[UploadFile] = File(None)):
    """Insert new conversation record and message record in configured database, returning AI Response"""
    conversation_schema = CreateConversationSchema(uuid=request.state.uuid)

    if (
        created_conversation_id := await ConversationRepo.create(conversation_schema=conversation_schema)
    ) is not None:
        data = { 'uuid': conversation_schema.uuid, 'conversation_id': created_conversation_id }
        if upload_file:
            await ingest_file(embedding_models, upload_file, data)
        message_schema = MessageSchema(type='human', content=content, conversation_id=created_conversation_id)     
        llm_stream = await chat(
            prompt_template, 
            models,
            embedding_models,
            data,
            message_schema)
        return StreamingResponse(llm_stream(), media_type="text/plain", headers={"X-Accel-Buffering": "no"})
        
    return {'error': f'Conversation not created'}, 400

@router.get(
    '/{id}',
    response_description="Get a single conversation",
    response_model=Union[ConversationSchema, dict],
    response_model_by_alias=False,
)
async def get_conversation(request: Request, id: str):
    """Get conversation record from configured database by id"""
    found_conversation = await ConversationRepo.find_by_aggregate(id, options={request.state.uuid_name: request.state.uuid})
    return found_conversation or {}

@router.put(
    "/{id}",
    response_description="Update a single conversation",
    response_model=Union[ConversationSchema, dict],
    response_model_by_alias=False,
)
async def update_conversation(request: Request, id: str, conversation_schema: UpdateConversationSchema = Body(...)):
    """Update individual fields of an existing conversation record and return modified fields to client."""
    if (
        updated_conversation := await ConversationRepo.update_one_and_return(
            id, 
            schema=conversation_schema, 
            options={ request.state.uuid_name: request.state.uuid} )
    ) is not None:
        return updated_conversation
    return {}
    
@router.delete(
    '/{id}', 
    response_description='Delete a conversation',
)
async def delete_conversation(request: Request, id: str):
    """Remove a single conversation record from the database."""
    if (
        delete_count := await ConversationRepo.delete(id, options={request.state.uuid_name: request.state.uuid})
    ) is not None:
        return {'delete_count': delete_count}  
    return {'delete_count': 0}

@router.delete(
    '/delete/all', 
    response_description='Delete all conversations',
)
async def delete_conversations(request: Request):
    """Remove a single conversation record from the database."""
    if (
        delete_count := await ConversationRepo.delete_many(options={request.state.uuid_name: request.state.uuid})
    ) is not None:
        return {'delete_count': delete_count}  
    return {'delete_count': 0}