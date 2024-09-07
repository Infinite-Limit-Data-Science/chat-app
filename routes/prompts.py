from fastapi import APIRouter, Request, status, Query, Body, Depends
from auth.bearer_authentication import get_current_user
from repositories.base_mongo_repository import base_mongo_factory as factory
from models.prompt import (
    PromptSchema, 
    PromptIdSchema,
    UpdatePromptSchema,
    Prompt
)

PromptRepo = factory(Prompt)

router = APIRouter(
    prefix='/prompts', 
    tags=['prompt'],
    dependencies=[Depends(get_current_user)]
)

@router.post(
    '/',
    response_description="Add new prompt",
    response_model=PromptIdSchema,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def create_prompt(request: Request, prompt_schema: PromptSchema = Body(...)):
    """Insert new prompt record in configured database, returning resource created"""
    if (
        created_prompt_id := await PromptRepo.create(request.state.uuid, prompt_schema)
    ) is not None:
        return { "_id": created_prompt_id }
    return {'error': f'Setting not created'}, 400

@router.get(
    '/{id}',
    response_description="Get a single prompt",
    response_model=PromptSchema,
    response_model_by_alias=False
)
async def get_prompt(request: Request, id: str):
    """Get prompt record from configured database by id"""
    if (
        prompt := await PromptRepo.find(request.state.uuid, id)
    ) is not None:
        return prompt
    return {'error': f'Prompt {id} not found'}, 404

@router.put(
    '/{id}',
    response_description="update a single prompt",
    response_model=PromptSchema,
    response_model_by_alias=False
)
async def update_prompt(request: Request, id: str, setting_schema: UpdatePromptSchema = Body(...)):
    """Get prompt record from configured database by id"""
    if (
        updated_prompt := await PromptRepo.update(request.state.uuid, id, setting_schema)
    ) is not None:
        return updated_prompt
    return {'error': f'Prompt {id} not found'}, 404

@router.delete(
    '/{id}', 
    response_description='Delete a prompt',
)
async def delete_prompt(request: Request, id: str):
    """Remove a single prompt record from the database."""
    if (
        deleted_prompt := await PromptRepo.delete(request.state.uuid, id)
    ) is not None:
        return deleted_prompt  
    return { 'error': f'Prompt {id} not found'}, 404