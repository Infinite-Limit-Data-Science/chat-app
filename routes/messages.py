import os
import shutil
from fastapi import APIRouter, status, Request, Body, Depends, File, UploadFile
from auth.bearer_authentication import get_current_user
from models.message import (
    MessageModel,
    MessageIdModel,
    UpdateMessageModel,
    MessageFacade as Message
)

router = APIRouter(
    prefix='/conversations', 
    tags=['conversation'],
    dependencies=[Depends(get_current_user)]
)

@router.post(
    '/{conversation_id}/message',
    response_description="Add new message",
    response_model=MessageIdModel,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
    tags=['message']
)
async def create_message(conversation_id: str, message: MessageModel = Body(...)):
    """Insert new message record in configured database, returning resource created"""
    if (
        created_message_id := await Message.create(conversation_id, message)
    ) is not None:
        return { "_id": created_message_id }
    return {'error': f'Conversation not created'}, 400

@router.get(
    '/{conversation_id}/message/{id}',
    response_description="Get a single message",
    response_model=MessageModel,
    response_model_by_alias=False,
    tags=['message']
)
async def get_message(request: Request, conversation_id: str, id: str):
    """Get message record from configured database by id"""
    if (
        message := await Message.find(request.state.uuid, conversation_id, id)
    ) is not None:
        return message
    return {'error': f'Message {id} not found'}, 404

@router.put(
    '/{conversation_id}/message/{id}',
    response_description="Update a single message",
    response_model=MessageModel,
    response_model_by_alias=False,
    tags=['message']
)
async def update_message(request: Request, conversation_id: str, id: str, message: UpdateMessageModel = Body(...)):
    """Update individual fields of an existing message record and return modified fields to client."""
    if (
        updated_message := await Message.update(request.state.uuid, conversation_id, id, message)
    ) is not None:
        return updated_message
    return {'error': f'Conversation {id} not found'}, 404
    
@router.delete(
    '/{conversation_id}/message/{id}', 
    response_description='Delete a message',
    tags=['message']
)
async def delete_message(request: Request, conversation_id: str, id: str):
    """Remove a single message record from the database."""
    if (
        deleted_message := await Message.delete(request.state.uuid, conversation_id, id)
    ) is not None:
        return deleted_message  
    return { 'error': f'Conversation {id} not found'}, 404

@router.post(
    '/{conversation_id}/message/{id}/upload',
    response_description='Upload a file',
    tags=['message']
)
async def upload(request: Request, conversation_id: str, id: str, upload_file: UploadFile = File(...)):
    path = f'files/{request.state.uuid}/conversations/{conversation_id}/messages/{id}/{upload_file.filename}'
    dir_path = os.path.dirname(path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(path, 'wb') as f:
       shutil.copyfileobj(upload_file.file, f)
    return { 'file': f'{upload_file.filename} uploaded successful' }