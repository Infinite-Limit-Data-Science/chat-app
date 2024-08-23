from fastapi import APIRouter, Query, Depends, logger
from auth.bearer_authentication import get_current_user
from models.user import UserModel

from models.conversation import (
    ConversationCollection,
    ConversationFacade as Conversation
)

router = APIRouter(
    prefix='/conversations', 
    tags=['dashbord']
)


@router.get(
    '/',
    response_description='List all conversations on Dashboard',
    response_model=ConversationCollection,
    response_model_by_alias=False,
)
async def conversations(record_offset: int = Query(0, description='record offset', alias='offset'), record_limit: int = Query(20, description="record limit", alias='limit'), user: UserModel = Depends(get_current_user)):
    """List conversations by an offset and limit per user"""
    return ConversationCollection(conversations=await Conversation.all(user.uuid, record_offset, record_limit))
