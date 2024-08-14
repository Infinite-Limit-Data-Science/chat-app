from fastapi import APIRouter
from typing import Optional

router = APIRouter()

@router.get('/conversations')
async def conversations(page: int = 1, page_size: Optional[int] = 20):
    return { 'message': f'{page_size} conversations on page {page}'}

@router.get('/conversations/new')
async def new_conversation():
    return { 'message': '' }

@router.post('/conversations/create')
async def create_conversation():
    return { 'message': '' }

@router.get('/conversations/{id}/show')
async def show_conversation(id: int):
    return { 'message': '' }

@router.get('/conversations/{id}/edit')
async def edit_conversation(id: int):
    return { 'message': '' }

@router.put('/conversations/{id}/update')
async def update_conversation(id: int):
    return { 'message': '' }

@router.delete('/conversations/{id}/delete')
async def delete_conversation(id: int):
    return { 'message': '' }
