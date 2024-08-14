from fastapi import APIRouter
from typing import Optional

router = APIRouter()

@router.get('/conversations/{id}/messages')
async def conversation_messages(id: int, page: int = 1, page_size: Optional[int] = 20):
    return { 'message': '' }

@router.get('/conversations/{id}/messages/new')
async def conversation_new_message(id: int):
    return { 'message': '' }

@router.post('/conversations/{id}/messages/create')
async def conversation_create_message(id: int):
    return { 'message': '' }

@router.get('/conversations/{id}/messages/{message_id}/show')
async def conversation_show_message(id: int, message_id: int):
    return { 'message': '' }

@router.get('/conversations/{id}/messages/{message_id}/edit')
async def conversation_edit_message(id: int, message_id: int):
    return { 'message': '' }

@router.put('/conversations/{id}/messages/{message_id}/update')
async def conversation_update_message(id: int, message_id: int):
    return { 'message': '' }

@router.delete('/conversations/{id}/messages/{message_id}/delete')
async def conversation_delete_message(id: int, message_id: int):
    return { 'message': '' }