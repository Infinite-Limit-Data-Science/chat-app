from fastapi import APIRouter, status, Request, Response, Query, Body, Depends
from auth.bearer_authentication import get_current_user
from models.conversation import (
    ConversationModel,
    CreateConversationModel,
    ConversationCollection, 
    UpdateConversationModel,
    ConversationIdModel,
    ConversationFacade as Conversation
)

router = APIRouter(
    prefix='/conversations', 
    tags=['conversation'],
    dependencies=[Depends(get_current_user)]
)

@router.get(
    '/',
    response_description='List all conversations',
    response_model=ConversationCollection,
    response_model_by_alias=False,
)
async def conversations(request: Request, record_offset: int = Query(0, description='record offset', alias='offset'), record_limit: int = Query(20, description="record limit", alias='limit')):
    """List conversations by an offset and limit"""    
    return ConversationCollection(conversations=await Conversation.all(request.state.uuid, record_offset, record_limit))

@router.post(
    '/',
    response_description="Add new conversation",
    response_model=ConversationIdModel,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def create_conversation(request: Request, response: Response, conversation: CreateConversationModel = Body(...)):
    """Insert new conversation record in configured database, returning resource created"""
    if (
        created_conversation_id := await Conversation.create(request.state.uuid, conversation)
    ) is not None:
        return { "_id": created_conversation_id }
    response.status_code = status.HTTP_400_BAD_REQUEST
    return {'error': f'Conversation not created'}

@router.get(
    '/{id}',
    response_description="Get a single conversation",
    response_model=ConversationModel,
    response_model_by_alias=False,
)
async def get_conversation(request: Request, id: str, response: Response):
    """Get conversation record from configured database by id"""
    if (
        conversation := await Conversation.find(request.state.uuid, id)
    ) is not None:
        return conversation
    response.status_code = status.HTTP_404_NOT_FOUND
    return {'error': f'Conversation {id} not found'}

@router.put(
    "/{id}",
    response_description="Update a single conversation",
    response_model=ConversationModel,
    response_model_by_alias=False,
)
async def update_conversation(request: Request, response: Response, id: str, conversation: UpdateConversationModel = Body(...)):
    """Update individual fields of an existing conversation record and return modified fields to client."""
    if (
        updated_conversation := await Conversation.update(request.state.uuid, id, conversation)
    ) is not None:
        return updated_conversation
    response.status_code = status.HTTP_404_NOT_FOUND
    return {'error': f'Conversation {id} not found'}
    
@router.delete(
    '/{id}', 
    response_description='Delete a conversation',
)
async def delete_conversation(request: Request, id: str, response: Response):
    """Remove a single conversation record from the database."""
    if (
        deleted_conversation := await Conversation.delete(request.state.uuid, id)
    ) is not None:
        return deleted_conversation  
    response.status_code = status.HTTP_404_NOT_FOUND
    return { 'error': f'Conversation {id} not found'}