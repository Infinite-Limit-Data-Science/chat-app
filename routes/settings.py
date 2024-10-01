import logging
from typing import Union
from fastapi import APIRouter, Request, status, Query, Body, Depends
from auth.bearer_authentication import get_current_user
from repositories.base_mongo_repository import base_mongo_factory as factory
from models.setting import (
    SettingSchema, 
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
    response_model=Union[SettingSchema, dict],
    response_model_by_alias=False,
)
async def get_setting(id: str):
    """Get setting record from configured database by id"""
    if (
        setting := await SettingRepo.find_one(id)
    ) is not None:
        return setting
    return {}

@router.put(
    '/{id}',
    response_description="update a single setting",
    response_model=Union[UpdateSettingSchema, dict],
    response_model_by_alias=False,
)
async def update_setting(id: str, setting_schema: UpdateSettingSchema = Body(...)):
    """Update settings, where associations are expected to be set up on client"""
    if (
        updated_setting := await SettingRepo.update_one(
            id, 
            _set=setting_schema.model_dump(by_alias=True))
    ) is not None:
        return updated_setting
    return {}