import logging
import os
import json
from typing import List, Dict, Optional
from fastapi import Request, Depends, HTTPException
from models.system_model_config import SystemModelConfigSchema
from models.setting import Setting, SettingSchema
from repositories.base_mongo_repository import base_mongo_factory as factory
from orchestrators.chat.llm_models.llm import LLM
from orchestrators.chat.llm_models.factories import FACTORIES as LLM_FACTORIES
from orchestrators.doc.embedding_models.factories import FACTORIES as EMBEDDING_FACTORIES
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from models.system_model_config import SystemModelConfigSchema
from models.user_model_config import UserModelConfigSchema
from models.system_embedding_config import SystemEmbeddingConfigSchema

_DEFAULT_PREPROMPT='You are an assistant for question-answering tasks. Answer the questions to the best of your ability.'

SettingRepo = factory(Setting)

async def get_user_settings(request: Request) -> SettingSchema:
    """Retrieve settings for current user"""
    setting = await SettingRepo.find_one(options={request.state.uuid_name: request.state.uuid})
    setting_schema = SettingSchema(**setting)
    return setting_schema

def load_system_model_config() -> Dict[str, SystemModelConfigSchema]:
    """Load system model configs from env"""
    model_dicts = json.loads(os.environ['MODELS'])
    system_configs = {config['name']: SystemModelConfigSchema(**config) for config in model_dicts}
    return system_configs

async def get_active_model_config(setting_id: str, configs: Dict[str, SystemModelConfigSchema], active_model_name: Optional[str]) -> SystemModelConfigSchema:
    """Get active model config or default to the first model config if none active"""
    if active_model_name:
        model_config = configs[active_model_name]
        active_model = active_model_name
    else:
        model_config = next(iter(configs.values()))
        active_model = model_config.name
    await SettingRepo.update_one(setting_id, _set={'activeModel': active_model})
    return model_config

async def refresh_model_configs(
    setting_schema: SettingSchema = Depends(get_user_settings),
    system_model_configs: Dict[str, SystemModelConfigSchema] = Depends(load_system_model_config)
) -> SystemModelConfigSchema:
    """Refresh model configs for the current user."""
    active_configs = {}
    active_model_name = None
    user_model_configs = {model_config.name: model_config for model_config in setting_schema.user_model_configs}

    for system_model_name, system_config in system_model_configs.items():
        user_config = user_model_configs.get(system_model_name)

        if user_config:
            system_config.parameters = user_config.parameters
            active_configs[system_model_name] = system_config

            if user_config.active:
                active_model_name = system_model_name
        else:
            active_configs[system_model_name] = system_config
            update_attributes = {
                'name': system_model_name,
                'parameters': system_config.parameters.model_dump(by_alias=True),
            }
            if not active_model_name and system_config.active:
                active_model_name = system_model_name
                update_attributes['active'] = True
            await SettingRepo.update_one(
                setting_schema.id,
                push={
                    'user_model_configs': UserModelConfigSchema(**update_attributes).model_dump(by_alias=True),
                }
            )

    return await get_active_model_config(setting_schema.id, active_configs, active_model_name)

async def get_current_models(
    request: Request,
    model_config_schema: SystemModelConfigSchema = Depends(refresh_model_configs)) -> List[LLM]:
    """Return the active model(s) of settings for current user"""
    models = [
        LLM_FACTORIES[endpoint['type']](**{
            'name': model_config_schema.name,
            'description': model_config_schema.description,
            'preprompt': model_config_schema.preprompt,
            'parameters': dict(model_config_schema.parameters),
            'server_kwargs': { 'headers': {'Authorization': f'Bearer {request.state.authorization}' }},
            'endpoint': endpoint,
        })
        for endpoint in model_config_schema.endpoints
    ]
    return models

async def get_current_embedding_models(request: Request) -> List[BaseEmbedding]:
    model_dict = json.loads(os.environ['EMBEDDING_MODELS'])
    model_configs = [SystemEmbeddingConfigSchema(**config) for config in model_dict]
    active_model_config = next((config for config in model_configs if config.active), None)
    if not active_model_config:
        HTTPException(status_code=404, detail='Expected Embedding Model, got None')
    models = [
        EMBEDDING_FACTORIES[endpoint['type']](**{
            'name': active_model_config.name,
            'description': active_model_config.description,
            'task': active_model_config.task,
            'endpoint': endpoint,
            'dimensions': active_model_config.dimensions,
            'token': request.state.authorization,
        })
        for endpoint in active_model_config.endpoints
    ]
    return models

async def get_prompt_template(
    setting_schema: SettingSchema = Depends(get_user_settings),
    system_config_schema: SystemModelConfigSchema = Depends(refresh_model_configs)) -> str:
    """Derive system prompt from either custom prompts or default system prompt"""
    if(
        active_user_config_id := next((config.id for config in setting_schema.user_model_configs if config.active), None)
    ) is not None:
        if(
            prompt := next((prompt for prompt in setting_schema.prompts for id in prompt.user_model_configs if id == active_user_config_id), None)
        ) is not None:
            return prompt.content
    
    return system_config_schema.preprompt or _DEFAULT_PREPROMPT