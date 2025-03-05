from fastapi import APIRouter, Request, Query, Depends
from ..logger import logger
from ..auth.bearer_authentication import get_current_user
from ..repositories.base_mongo_repository import base_mongo_factory as factory
from ..repositories.conversation_mongo_repository import ConversationMongoRepository
from ..models.conversation import Conversation, ConversationSchema, ConversationCollectionSchema
from ..models.setting import Setting, SettingSchema

ConversationRepo = factory(Conversation)

SettingRepo = factory(Setting)

router = APIRouter(
    tags=['dashbord'],
    dependencies=[Depends(get_current_user)],
)

@router.get(
    '/dashboard',
    response_description='Load dashbord items, including user settings and conversations',
)
async def dashboard(
    request: Request, 
    record_offset: int = Query(0, description='record offset', alias='offset'), 
    record_limit: int = Query(20, description="record limit", alias='limit', le=10000)):
    """Load dashboard"""
    logger.info('Loading Dashboard')

    setting_attributes = await SettingRepo.find_one(options={ 'sessionId': request.state.uuid })
    settings = SettingSchema(**setting_attributes)
    conversation_attributes = await ConversationMongoRepository.all(
        options={request.state.uuid_name: request.state.uuid}, offset=record_offset, limit=record_limit)
    conversations = [ConversationSchema(**attributes) for attributes in conversation_attributes]
    conversation_collection = ConversationCollectionSchema(conversations=conversations)
    
    return {
        'uuid': request.state.uuid,
        'settings': settings,
        'conversations': conversation_collection,
        'session_id': request.state.session_id,
    }