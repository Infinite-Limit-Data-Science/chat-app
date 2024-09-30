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
async def get_setting(request: Request, id: str):
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
    # TODO: make this serializer; also prompt id not being saved
    from bson import ObjectId
    user_model_configs = setting_schema.user_model_configs
    prompts = setting_schema.prompts

    updated_user_model_configs = []
    for config in user_model_configs:
        if isinstance(config.id, str):
            config.id = ObjectId(config.id)
        config.prompts = [ObjectId(prompt_id) for prompt_id in config.prompts if isinstance(prompt_id, str)]
        updated_user_model_configs.append(config)
    setting_schema.user_model_configs = updated_user_model_configs
        
    updated_prompts = []
    for prompt in prompts:
        if isinstance(prompt.id, str):
            prompt.id = ObjectId(prompt.id)
        
        prompt.user_model_configs = [ObjectId(config_id) for config_id in prompt.user_model_configs if isinstance(config_id, str)]

        for config in setting_schema.user_model_configs:
            if config.id in prompt.user_model_configs and prompt.id not in config.prompts:
                config.prompts.append(prompt.id)

        updated_prompts.append(prompt)

    for config in setting_schema.user_model_configs:
        for prompt_id in config.prompts:
            for prompt in updated_prompts:
                if prompt.id == prompt_id and config.id not in prompt.user_model_configs:
                    prompt.user_model_configs.append(config.id)
    
    if (
        updated_setting := await SettingRepo.update_one(
            id, 
            _set=setting_schema.model_dump(by_alias=True, exclude_unset=True))
    ) is not None:
        return updated_setting
    return {}