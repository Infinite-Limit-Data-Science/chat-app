from fastapi import APIRouter, status, Body, HTTPException
from models.conversation import (
    ConversationModel, 
    ConversationCollection, 
    UpdateConversationModel, 
    ConversationFacade as Conversation
)

router = APIRouter()

@router.get(
    '/conversations',
    response_description='List all conversations',
    response_model=ConversationCollection,
    response_model_by_alias=False
)
async def conversations(page: int = 1, page_size: int = 20):
    """List conversations"""
    return ConversationCollection(conversations=await Conversation.all(page, page_size))

@router.post(
    '/conversations/create',
    response_description="Add new conversation",
    response_model=ConversationModel,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def create_conversation(conversation: ConversationModel = Body(...)):
    """Insert new conversation record"""
    return await Conversation.create(conversation)

@router.get(
    '/conversations/{id}',
    response_description="Get a single conversation",
    response_model=ConversationModel,
    response_model_by_alias=False,
)
async def update_conversation(id: str):
    """Get the record for a specific conversation"""
    if (
        conversation := await Conversation.find(id)
    ) is not None:
        return conversation
    raise HTTPException(status_code=404, detail=f"Conversation {id} not found")

@router.put(
    "/conversations/{id}",
    response_description="Update a conversation",
    response_model=ConversationModel,
    response_model_by_alias=False,
)
async def update_conversation(id: int, conversation: UpdateConversationModel = Body(...)):
    """Update individual fields of an existing conversation record."""
    conversation = await Conversation.update(id, conversation)
    if not conversation:
        raise HTTPException(status_code=404, detail=f"Conversation {id} not found")
    return conversation
    

@router.delete('/conversations/{id}/delete')
async def delete_conversation(id: int):
    pass
    # NOT YET IMPLEMENTED