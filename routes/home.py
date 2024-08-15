from fastapi import APIRouter
from models.conversation import ConversationFacade as Conversation

router = APIRouter()

@router.get('/')
async def home():
    conversations = await Conversation.all(1, 20)
    return conversations