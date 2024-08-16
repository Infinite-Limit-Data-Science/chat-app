from typing import Optional
from fastapi import APIRouter, status, Response, Query, Body
from models.conversation import (
    ConversationModel, 
    ConversationCollection, 
    UpdateConversationModel, 
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
async def conversations(page: int = Query(1, description="record offset", alias='offset'), page_size: int = Query(20, description="record limit", alias='limit')):
    """List conversations paginated by a page and page size integer"""
    return ConversationCollection(conversations=await Conversation.all(page, page_size))

@router.post(
    '/',
    response_description="Add new conversation",
    response_model=ConversationModel,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
    tags=['conversation']
)
async def create_conversation(conversation: ConversationModel = Body(...)):
    """Insert new conversation record in configured database, returning resource created"""
    return await Conversation.create(conversation)

@router.get(
    '/{id}',
    response_description="Get a single conversation",
    response_model=ConversationModel,
    response_model_by_alias=False,
    tags=['conversation']
)
async def get_conversation(id: str, response: Response):
    """Get conversation record from configured database by conversation id"""
    if (
        conversation := await Conversation.find(id)
    ) is not None:
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
async def update_conversation(id: int, response: Response, conversation: UpdateConversationModel = Body(...)):
    """Update individual fields of an existing conversation record and return modified fields to client."""
    conversation: Optional[ConversationModel] = await Conversation.update(id, conversation)
    if conversation is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'error': f'Conversation {id} not found'}
    return conversation
    
@router.delete(
    '/{id}', 
    response_description='Delete a conversation',
    tags=['conversation']
)
async def delete_conversation(id: int, response: Response):
    """Remove a single conversation record from the database."""
    deleted = await Conversation.delete(id)
    if deleted:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    response.status_code = status.HTTP_404_NOT_FOUND
    return { 'error': f'Conversation {id} not found'}