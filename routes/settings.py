from fastapi import APIRouter, Request, status, Query, Body, Depends
from auth.bearer_authentication import get_current_user
from repositories.base_mongo_repository import base_mongo_factory as factory
from models.setting import (
    SettingSchema, 
    SettingIdSchema,
    UpdateSettingSchema,
    Setting
)

SettingRepo = factory(Setting)

router = APIRouter(
    prefix='/settings', 
    tags=['setting'],
    dependencies=[Depends(get_current_user)]
)

@router.post(
    '/',
    response_description="Add new setting",
    response_model=SettingIdSchema,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def create_setting(request: Request, setting_schema: SettingSchema = Body(...)):
    """Insert new setting record in configured database, returning resource created"""
    if (
        created_setting_id := await SettingRepo.create(request.state.uuid, setting_schema)
    ) is not None:
        return { "_id": created_setting_id }
    return {'error': f'Setting not created'}, 400

@router.get(
    '/{id}',
    response_description="Get a single setting",
    response_model=SettingSchema,
    response_model_by_alias=False,
    tags=['setting']
)
async def get_setting(request: Request, id: str):
    """Get setting record from configured database by id"""
    if (
        setting := await SettingRepo.find(request.state.uuid, id)
    ) is not None:
        return setting
    return {'error': f'Setting {id} not found'}, 404

@router.put(
    '/{id}',
    response_description="update a single setting",
    response_model=SettingSchema,
    response_model_by_alias=False,
    tags=['setting']
)
async def update_setting(request: Request, id: str, setting_schema: UpdateSettingSchema = Body(...)):
    """Get setting record from configured database by id"""
    if (
        updated_setting := await SettingRepo.update(request.state.uuid, id, setting_schema)
    ) is not None:
        return updated_setting
    return {'error': f'Setting {id} not found'}, 404

@router.delete(
    '/{id}', 
    response_description='Delete a conversation',
)
async def delete_setting(request: Request, id: str):
    """Remove a single conversation record from the database."""
    if (
        deleted_conversation := await SettingRepo.delete(request.state.uuid, id)
    ) is not None:
        return deleted_conversation  
    return { 'error': f'Conversation {id} not found'}, 404