from fastapi import APIRouter, status, Response, Query, Body
from models.setting import (
    SettingModel, 
    SettingIdModel,
    SettingFacade as Setting
)

router = APIRouter(
    prefix='/settings', tags=['setting']
)

@router.post(
    '/',
    response_description="Add new setting",
    response_model=SettingIdModel,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def create_setting(response: Response, setting: SettingModel = Body(...)):
    """Insert new setting record in configured database, returning resource created"""
    if (
        created_setting_id := await Setting.create(setting)
    ) is not None:
        return { "_id": created_setting_id }
    response.status_code = status.HTTP_400_BAD_REQUEST
    return {'error': f'Setting not created'}

@router.get(
    '/{id}',
    response_description="Get a single setting",
    response_model=SettingModel,
    response_model_by_alias=False,
    tags=['setting']
)
async def get_setting(id: str, response: Response):
    """Get setting record from configured database by id"""
    if (
        setting := await Setting.find(id)
    ) is not None:
        return setting
    response.status_code = status.HTTP_404_NOT_FOUND
    return {'error': f'Setting {id} not found'}