from fastapi import APIRouter, status, Response, Query, Body, logger
from typing import Optional
from models.conversation import (
    ConversationFacade as Conversation
)
from models.message import (
    MessageModel,
    MessageCollection,
    MessageIdModel,
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

# @router.get(
#     '/{id}',
#     response_description="Get a single conversation",
#     response_model=ConversationModel,
#     response_model_by_alias=False,
#     tags=['conversation']
# )
# async def get_message(id: str, response: Response):
#     """Get conversation record from configured database by session id"""
#     if (
#         conversation := await Conversation.find(id)
#     ) is not None:
#         # ConversationModel(**conversation)
#         return conversation
#     response.status_code = status.HTTP_404_NOT_FOUND
#     return {'error': f'Conversation {id} not found'}

# @router.put(
#     "/{id}",
#     response_description="Update a single conversation",
#     response_model=ConversationModel,
#     response_model_by_alias=False,
#     tags=['conversation']
# )
# async def update_message(id: str, response: Response, conversation: UpdateConversationModel = Body(...)):
#     """Update individual fields of an existing conversation record and return modified fields to client."""
#     if (
#         updated_conversation := await Conversation.update(id, conversation)
#     ) is not None:
#         return updated_conversation
#     response.status_code = status.HTTP_404_NOT_FOUND
#     return {'error': f'Conversation {id} not found'}
    
# @router.delete(
#     '/{id}', 
#     response_description='Delete a conversation',
#     tags=['conversation']
# )
# async def delete_message(id: str, response: Response):
#     """Remove a single conversation record from the database."""
#     if (
#         deleted_conversation := await Conversation.delete(id)
#     ) is not None:
#         return deleted_conversation  
#     response.status_code = status.HTTP_404_NOT_FOUND
#     return { 'error': f'Conversation {id} not found'}