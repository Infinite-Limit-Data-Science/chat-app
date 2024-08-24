from fastapi import APIRouter, Request, Response, status, Query, Body, Depends
from auth.bearer_authentication import get_current_user
from models.setting import (
    SettingModel, 
    SettingIdModel,
    UpdateSettingModel,
    SettingFacade as Setting
)

router = APIRouter(
    prefix='/settings', 
    tags=['setting'],
    dependencies=[Depends(get_current_user)]
)

@router.post(
    '/',
    response_description="Add new setting",
    response_model=SettingIdModel,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def create_setting(request: Request, setting: SettingModel = Body(...)):
    """Insert new setting record in configured database, returning resource created"""
    if (
        created_setting_id := await Setting.create(request.state.uuid, setting)
    ) is not None:
        return { "_id": created_setting_id }
    return {'error': f'Setting not created'}, 400

@router.get(
    '/{id}',
    response_description="Get a single setting",
    response_model=SettingModel,
    response_model_by_alias=False,
    tags=['setting']
)
async def get_setting(request: Request, id: str):
    """Get setting record from configured database by id"""
    if (
        setting := await Setting.find(request.state.uuid, id)
    ) is not None:
        return setting
    return {'error': f'Setting {id} not found'}, 404

@router.put(
    '/{id}',
    response_description="update a single setting",
    response_model=SettingModel,
    response_model_by_alias=False,
    tags=['setting']
)
async def update_setting(request: Request, id: str, setting: UpdateSettingModel = Body(...)):
    """Get setting record from configured database by id"""
    if (
        setting := await Setting.update(request.state.uuid, id, setting)
    ) is not None:
        return setting
    return {'error': f'Setting {id} not found'}, 404