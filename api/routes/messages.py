from typing import Optional, List, Union
from bson import ObjectId
from fastapi import (
    APIRouter, status, Request, Form, 
    Depends, File, UploadFile)
from fastapi.responses import StreamingResponse
from ..langchain_chat import LLM
from ..langchain_doc import BaseEmbedding
from ..logger import logger
from ..models.mongo_schema import ObjectId
from ..auth.bearer_authentication import get_current_user
from .chats import chat
from .configs import (
    get_current_models, get_current_embedding_models, 
    get_prompt_template, get_current_guardrails,
    DEFAULT_PREPROMPT)
from .uploads import ingest_files
from ..models.message import (
    Message,
    MessageSchema,
)
from ..repositories.base_mongo_repository import (
    base_mongo_factory as factory)
from ..repositories.conversation_mongo_repository import (
    ConversationMongoRepository as ConversationRepo)

# TODO: extract to env var
_DATABASE_STRATEGY = 'mongodb'

MessageRepo = factory(Message)

router = APIRouter(
    prefix='/conversations', 
    tags=['conversation'],
    dependencies=[Depends(get_current_user)]
)

@router.post(
    '/{conversation_id}/message',
    response_description="Add new message",
    status_code=status.HTTP_201_CREATED,
    tags=['message']
)
async def create_message(
    request: Request,
    conversation_id: str,
    content: str = Form(...),
    upload_files: Optional[List[UploadFile]] = File(None),
    models: List[LLM] = Depends(get_current_models),
    embedding_models: List[BaseEmbedding]  = Depends(get_current_embedding_models),
    guardrails: List[LLM] = Depends(get_current_guardrails),
    prompt_template: str = Depends(get_prompt_template)):
    """Insert new message record in configured database, returning AI Response"""
    logger.info(f'invoking message endpoint with content `{content}`')

    retrievers = []
    if _DATABASE_STRATEGY == 'mongodb':
        conversation_id = ObjectId(conversation_id)
    data = { 'uuid': request.state.uuid, 'conversation_id': conversation_id }
    if upload_files:
        retrievers, filenames = await ingest_files(embedding_models, upload_files, data)
        await ConversationRepo.update_one(conversation_id, _set={ 'filenames': filenames })
    message_schema = MessageSchema(type='human', content=content, conversation_id=conversation_id)
    prompt = prompt_template or DEFAULT_PREPROMPT

    llm_stream = await chat(
        prompt, 
        models, 
        guardrails, 
        embedding_models, 
        data, 
        retrievers, 
        message_schema)
    
    return StreamingResponse(llm_stream(), media_type='text/event-stream', headers={'X-Accel-Buffering': 'no'})

@router.get(
    '/{conversation_id}/message/{id}',
    response_description="Get a single message",
    response_model=Union[MessageSchema, dict],
    response_model_by_alias=False,
    tags=['message']
)
async def get_message(request: Request, conversation_id: str, id: str):
    """Get message record from configured database by id"""
    if (
        message := await MessageRepo.find_one(id, options={'conversation_id': ObjectId(conversation_id) })
    ) is not None:
        return message
    
    return {}

# Note Conversational AI does not allow the edit of existing messages
@router.delete(
    '/{conversation_id}/message/{id}', 
    response_description='Delete a message',
    tags=['message']
)
async def delete_message(request: Request, conversation_id: str, id: str):
    """Remove a single message record from the database."""
    if (
        delete_count := await MessageRepo.delete(id, options={'conversation_id': ObjectId(conversation_id)})
    ) is not None:
        await ConversationRepo.remove_from_field(conversation_id, options={'message_ids': ObjectId(id) })
        return {'delete_count': delete_count} 
     
    return {'delete_count': 0}