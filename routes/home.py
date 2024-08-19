from fastapi import APIRouter, Query
from models.conversation import (
    ConversationCollection,
    ConversationFacade as Conversation
)

router = APIRouter()

@router.get(
    '/',
    response_description='List all conversations on Dashboard',
    response_model=ConversationCollection,
    response_model_by_alias=False,
    tags=['conversation']
)
async def conversations(record_offset: int = Query(0, description='record offset', alias='offset'), record_limit: int = Query(20, description="record limit", alias='limit')):
    """List conversations by an offset and limit"""
    return ConversationCollection(conversations=await Conversation.all(record_offset, record_limit))
