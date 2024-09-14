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

@router.get(
    '/{id}',
    response_description="Get a single setting",
    response_model=SettingSchema,
    response_model_by_alias=False,
)
async def get_setting(request: Request, id: str):
    """Get setting record from configured database by id"""
    if (
        setting := await SettingRepo.find(id)
    ) is not None:
        return setting
    return {'error': f'Setting {id} not found'}, 404

# TODO: update this below
@router.put(
    '/{id}',
    response_description="update a single setting",
    response_model=SettingSchema,
    response_model_by_alias=False,
)
async def update_setting(request: Request, id: str, setting_schema: UpdateSettingSchema = Body(...)):
    """Get setting record from configured database by id"""
    if (
        updated_setting := await SettingRepo.update(request.state.uuid, id, setting_schema)
    ) is not None:
        return updated_setting
    return {'error': f'Setting {id} not found'}, 404