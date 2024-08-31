from fastapi import APIRouter, Query, Depends, logger
from auth.bearer_authentication import get_current_user
from models.user import UserSchema
from models.conversation import (
    ConversationCollectionSchema,
    Conversation
)
from repositories.base_mongo_repository import base_mongo_factory as factory

router = APIRouter(
    tags=['dashbord']
)

@router.get(
    '/',
    response_description='List all conversations on Dashboard',
    response_model=ConversationCollectionSchema,
    response_model_by_alias=False,
)
async def conversations(record_offset: int = Query(0, description='record offset', alias='offset'), record_limit: int = Query(20, description="record limit", alias='limit'), user: UserSchema = Depends(get_current_user)):
    """List conversations by an offset and limit per user"""
    ConversationRepo = factory(Conversation)
    return ConversationCollectionSchema(conversations=await ConversationRepo.all(user.uuid, record_offset, record_limit))
