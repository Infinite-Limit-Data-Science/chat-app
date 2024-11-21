from typing import List, Optional, Union
from fastapi import ( 
    APIRouter, status, Request, Query, Body, Form, 
    Depends, File, UploadFile)
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse
from huggingface_hub.errors import HfHubHTTPError
from ..langchain_chat import LLM
from ..langchain_doc import BaseEmbedding
from ..logger import logger
from ..auth.bearer_authentication import get_current_user
from .chats import chat
from .configs import (
    get_current_models, get_current_embedding_models, 
    get_prompt_template, get_current_guardrails,
    DEFAULT_PREPROMPT)
from .uploads import ingest_files
from ..repositories.conversation_mongo_repository import ( 
    ConversationMongoRepository as ConversationRepo)
from ..models.conversation import (
    ConversationSchema,
    CreateConversationSchema,
    ConversationCollectionSchema, 
    UpdateConversationSchema,
)
from ..models.message import MessageSchema

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
async def conversations(
        request: Request, 
        record_offset: int = Query(0, description='record offset', alias='offset'), 
        record_limit: int = Query(20, description="record limit", alias='limit'),
        sort: str = Query(None, description='sort field and direction as sort=field[asc|desc]', alias='sort')
    ):
    """List conversations by an offset and limit"""
    logger.info('loading conversations')

    sort_field, sort_direction = 'updatedAt', 'desc'
    if sort:
        parts = sort.split('[')
        if len(parts) == 2:
            sort_field = parts[0]
            sort_direction = parts[1].strip(']')
            if sort_direction not in ['asc', 'desc']:
                sort_direction = 'desc'

    conversations = await ConversationRepo.all(
        options={
            request.state.uuid_name: request.state.uuid
        }, 
        offset=record_offset, 
        limit=record_limit,
        sort_field=sort_field,
        sort_direction=sort_direction)
    return ConversationCollectionSchema(conversations=conversations)

@router.post(
    '/',
    response_description="Add new conversation",
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    request: Request,
    content: str = Form(...),
    upload_files: Optional[List[UploadFile]] = File(None),
    models: List[LLM] = Depends(get_current_models),
    embedding_models: List[BaseEmbedding] = Depends(get_current_embedding_models),
    guardrails: Optional[List[LLM]] = Depends(get_current_guardrails),
    prompt_template: str = Depends(get_prompt_template),
):
    """Insert new conversation record and message record in configured database, returning AI Response"""
    logger.info(f'invoking conversation endpoint with content `{content}`')

    conversation_schema = CreateConversationSchema(
        uuid=request.state.uuid, 
        model_name=models[0].name, 
        prompt_used=prompt_template)
    retrievers = []
    
    if (
        created_conversation_id := await ConversationRepo.create(conversation_schema=conversation_schema)
    ) is not None:
        data = { 'uuid': conversation_schema.uuid, 'conversation_id': created_conversation_id }
        if upload_files:
            retrievers, filenames = await ingest_files(embedding_models, upload_files, data)
            await ConversationRepo.update_one(created_conversation_id, _set={ 'filenames': filenames })

        message_schema = MessageSchema(
            type='human', 
            content=content, 
            conversation_id=created_conversation_id) 
        prompt = prompt_template or DEFAULT_PREPROMPT

        try:    
            llm_stream = await chat(
                prompt,
                models, 
                guardrails, 
                embedding_models, 
                data, 
                retrievers, 
                message_schema) 
            return StreamingResponse(llm_stream(), media_type='text/event-stream')
        except HfHubHTTPError as e:
            error_info = {
                'url': e.response.url,
                'status_code': e.response.status_code,
                'error_message': e.response.text,
                'error_type': type(e).__name__,
            }
            logger.warning(f'Request failed error_info {error_info}')
            await ConversationRepo.delete(created_conversation_id, options={ request.status.uuid_name: request.status.uuid})
            raise HTTPException(status_code=e.response.status_code, detail=error_info)
        
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
    response_model=Union[UpdateConversationSchema, dict],
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
        return UpdateConversationSchema(**updated_conversation)
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
    """Remove all conversation records from the database for given user."""
    if (
        delete_count := await ConversationRepo.delete_many(options={request.state.uuid_name: request.state.uuid})
    ) is not None:
        return {'delete_count': delete_count}  
    return {'delete_count': 0}