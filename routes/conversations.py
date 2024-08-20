from fastapi import APIRouter, status, Response, Query, Body
from models.conversation import (
    ConversationModel, 
    ConversationCollection, 
    UpdateConversationModel,
    ConversationIdModel,
    ConversationFacade as Conversation
)

router = APIRouter(prefix='/conversations', tags=['conversation'])

@router.get(
    '/',
    response_description='List all conversations',
    response_model=ConversationCollection,
    response_model_by_alias=False,
    tags=['conversation']
)
async def conversations(record_offset: int = Query(0, description='record offset', alias='offset'), record_limit: int = Query(20, description="record limit", alias='limit')):
    """List conversations by an offset and limit"""
    return ConversationCollection(conversations=await Conversation.all(record_offset, record_limit))

@router.post(
    '/',
    response_description="Add new conversation",
    response_model=ConversationIdModel,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
    tags=['conversation']
)
async def create_conversation(response: Response, conversation: ConversationModel = Body(...)):
    """Insert new conversation record in configured database, returning resource created"""
    if (
        created_conversation_id := await Conversation.create(conversation)
    ) is not None:
        return { "_id": created_conversation_id }
    response.status_code = status.HTTP_400_BAD_REQUEST
    return {'error': f'Conversation not created'}

@router.get(
    '/{id}',
    response_description="Get a single conversation",
    response_model=ConversationModel,
    response_model_by_alias=False,
    tags=['conversation']
)
async def get_conversation(id: str, response: Response):
    """Get conversation record from configured database by id"""
    if (
        conversation := await Conversation.find(id)
    ) is not None:
        # ConversationModel(**conversation)
        return conversation
    response.status_code = status.HTTP_404_NOT_FOUND
    return {'error': f'Conversation {id} not found'}

@router.put(
    "/{id}",
    response_description="Update a single conversation",
    response_model=ConversationModel,
    response_model_by_alias=False,
    tags=['conversation']
)
async def update_conversation(id: str, response: Response, conversation: UpdateConversationModel = Body(...)):
    """Update individual fields of an existing conversation record and return modified fields to client."""
    if (
        updated_conversation := await Conversation.update(id, conversation)
    ) is not None:
        return updated_conversation
    response.status_code = status.HTTP_404_NOT_FOUND
    return {'error': f'Conversation {id} not found'}
    
@router.delete(
    '/{id}', 
    response_description='Delete a conversation',
    tags=['conversation']
)
async def delete_conversation(id: str, response: Response):
    """Remove a single conversation record from the database."""
    if (
        deleted_conversation := await Conversation.delete(id)
    ) is not None:
        return deleted_conversation  
    response.status_code = status.HTTP_404_NOT_FOUND
    return { 'error': f'Conversation {id} not found'}