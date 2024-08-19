from fastapi import APIRouter
from models.conversation import (
    ConversationCollection,
    ConversationFacade as Conversation
)

router = APIRouter()

@router.get(
    '/',
    response_description='List all conversations on dashboard',
    response_model=ConversationCollection,
    response_model_by_alias=False,
)
async def home():
    """List conversations"""
    return ConversationCollection(conversations=await Conversation.all(1, 20))