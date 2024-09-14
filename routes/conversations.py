from typing import List
from fastapi import APIRouter, status, Request, Query, Body, Depends, logger
from auth.bearer_authentication import get_current_user
from models.mongo_schema import ObjectId
from routes.chats import get_current_models, get_system_prompt, get_message_history, chat
from orchestrators.chat.llm_models.llm import LLM
from orchestrators.chat.messages.message_history import MongoMessageHistory
from models.llm_schema import PromptDict
from repositories.conversation_mongo_repository import ConversationMongoRepository as ConversationRepo
from models.conversation import (
    ConversationSchema,
    ConversationCollectionSchema, 
    UpdateConversationSchema,
    ConversationIdSchema,
)
from models.message import MessageSchema

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
    return ConversationCollectionSchema(conversations=await ConversationRepo.all(options={request.state.uuid_name: request.state.uuid}, offset=record_offset, limit=record_limit))

@router.post(
    '/',
    response_description="Add new conversation",
    response_model=ConversationIdSchema,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)

async def create_conversation(
    request: Request, 
    conversation: ConversationSchema = Body(...),
    messages: List[MessageSchema] = Body(...),
    models: List[LLM] = Depends(get_current_models), 
    system_prompt: str = Depends(get_system_prompt),
    mongo_message_history: MongoMessageHistory = Depends(get_message_history)):
    """Insert new conversation record and message record in configured database, returning AI Response"""
    conversation.uuid = request.state.uuid
    if (
        created_conversation_id := await ConversationRepo.create(conversation_schema=conversation, messages_schema=messages)
    ) is not None:
        logger.logging.warning(f'CREATED CONV ID {created_conversation_id}')
        ai_message = await chat(
            system_prompt, 
            models, 
            mongo_message_history, 
            { 'conversation_id': created_conversation_id },
            messages[0])
        logger.logging.warning(f'AI Messages: {ai_message}')
        
    return {'error': f'Conversation not created'}, 400

@router.get(
    '/{id}',
    response_description="Get a single conversation",
    response_model=ConversationSchema,
    response_model_by_alias=False,
)
async def get_conversation(request: Request, id: str):
    """Get conversation record from configured database by id"""
    if (
        found_conversation := await ConversationRepo.find_one(id, options={request.state.uuid_name: request.state.uuid})
    ) is not None:
        return found_conversation
    return {'error': f'Conversation {id} not found'}, 404

@router.put(
    "/{id}",
    response_description="Update a single conversation",
    response_model=ConversationSchema,
    response_model_by_alias=False,
)
async def update_conversation(request: Request, id: str, conversation_schema: UpdateConversationSchema = Body(...)):
    """Update individual fields of an existing conversation record and return modified fields to client."""
    if (
        updated_conversation := await ConversationRepo.update_one(options={'_id': ObjectId(id), request.state.uuid_name: request.state.uuid}, assigns=dict(conversation_schema))
    ) is not None:
        return updated_conversation
    return {'error': f'Conversation {id} not found'}, 404
    
@router.delete(
    '/{id}', 
    response_description='Delete a conversation',
)
async def delete_conversation(request: Request, id: str):
    """Remove a single conversation record from the database."""
    if (
        deleted_conversation := await ConversationRepo.delete(id, options={request.state.uuid_name: request.state.uuid})
    ) is not None:
        return deleted_conversation  
    return { 'error': f'Conversation {id} not found'}, 404