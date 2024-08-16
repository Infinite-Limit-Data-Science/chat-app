from fastapi import APIRouter, status, Path, Query, Body
from typing import Optional
from models.conversation import (
    ConversationFacade as Conversation
)
from models.message import (
    MessageModel,
    MessageCollection,
    MessageFacade as Message
)

router = APIRouter(prefix='/conversations', tags=['conversation'])

@router.get(
    '/{id}/messages',
    response_description="Get all messages per conversation",
    response_model=MessageCollection,
    response_model_by_alias=False,
    tags=['message']
)
async def conversation_messages(id: int = Path(description='conversation id'), page: int = Query(1, description="record offset", alias='offset'), page_size: int = Query(20, description="record limit", alias='limit')):
    return await Message.find(id)

@router.post(
    '/{id}/message',
    response_description="Add new message",
    response_model=MessageModel,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
    tags=['message'],
)
async def create_message(id: str = Path(description='conversation id'), message: MessageModel = Body(...)):
    """Insert new message record in configured database, returning resource created"""
    return await Message.create(message)