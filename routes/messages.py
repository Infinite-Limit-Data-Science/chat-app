from fastapi import APIRouter, status, Response, Query, Body
from models.message import (
    MessageModel,
    MessageIdModel,
    UpdateMessageModel,
    MessageFacade as Message
)

router = APIRouter(prefix='/conversations', tags=['conversation'])

@router.post(
    '/{conversation_id}/message',
    response_description="Add new message",
    response_model=MessageIdModel,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
    tags=['message']
)
async def create_message(response: Response, conversation_id: str, message: MessageModel = Body(...)):
    """Insert new message record in configured database, returning resource created"""
    if (
        created_message_id := await Message.create(conversation_id, message)
    ) is not None:
        return { "_id": created_message_id }
    response.status_code = status.HTTP_400_BAD_REQUEST
    return {'error': f'Conversation not created'}

@router.get(
    '/{conversation_id}/message/{id}',
    response_description="Get a single message",
    response_model=MessageModel,
    response_model_by_alias=False,
    tags=['message']
)
async def get_message(conversation_id: str, id: str, response: Response):
    """Get message record from configured database by id"""
    if (
        message := await Message.find(conversation_id, id)
    ) is not None:
        return message
    response.status_code = status.HTTP_404_NOT_FOUND
    return {'error': f'Message {id} not found'}

@router.put(
    '/{conversation_id}/message/{id}',
    response_description="Update a single message",
    response_model=MessageModel,
    response_model_by_alias=False,
    tags=['message']
)
async def update_message(conversation_id: str, id: str, response: Response, message: UpdateMessageModel = Body(...)):
    """Update individual fields of an existing message record and return modified fields to client."""
    if (
        updated_message := await Message.update(conversation_id, id, message)
    ) is not None:
        return updated_message
    response.status_code = status.HTTP_404_NOT_FOUND
    return {'error': f'Conversation {id} not found'}
    
@router.delete(
    '/{conversation_id}/message/{id}', 
    response_description='Delete a message',
    tags=['message']
)
async def delete_message(conversation_id: str, id: str, response: Response):
    """Remove a single message record from the database."""
    if (
        deleted_message := await Message.delete(conversation_id, id)
    ) is not None:
        return deleted_message  
    response.status_code = status.HTTP_404_NOT_FOUND
    return { 'error': f'Conversation {id} not found'}