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
from models.model_observer import ModelSubject, ModelObserver
from models.classification import get_strategy_for_classification

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

async def get_active_model_config(
    setting_id: str, 
    configs: Dict[str, SystemModelConfigSchema], 
    active_model_name: Optional[str]
) -> SystemModelConfigSchema:
    """Get active model config or default to the first model config if none active"""
    if active_model_name:
        model_config = next(iter(configs.values()))
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
    classification_subjects = {}
    user_model_configs = {model_config.name: model_config for model_config in setting_schema.user_model_configs}
    active_model_name = None

    for system_model_name, system_config in system_model_configs.items():
        classification = system_config.classification
        
        strategy = get_strategy_for_classification(classification)
        
        if not strategy.is_text_generation():
            continue
        
        if classification not in classification_subjects:
            classification_subjects[classification] = ModelSubject(classification)

        observer = ModelObserver(name=system_model_name)
        classification_subjects[classification].add_observer(observer)

        if system_model_name in user_model_configs:
            user_config = user_model_configs[system_model_name]
            system_config.parameters = user_config.parameters 
        else:
            new_user_model_config = UserModelConfigSchema(
                name=system_model_name,
                description=system_config.description,
                preprompt=system_config.preprompt,
                parameters=system_config.parameters,
                classification=system_config.classification,
                active=system_config.active
            )

            await SettingRepo.update_one(
                setting_schema.id,
                push={
                    'user_model_configs': new_user_model_config.model_dump(by_alias=True),
                }
            )

            user_model_configs[system_model_name] = new_user_model_config

    for classification, subject in classification_subjects.items():
        strategy = get_strategy_for_classification(classification)
        
        if not strategy.is_text_generation(): # and not strategy.is_image_to_text()
            continue
        
        # TODO: update existing user_model_configs in db with classification attribute
        for user_model_name, user_model_config in user_model_configs.items():
            if user_model_config.classification == classification:
                if user_model_config.active:
                    subject.set_active(user_model_name)
                    active_model_name = user_model_name
                    break
        
        if not active_model_name:
            for system_model_name in subject.observers:
                strategy.set_active(system_model_configs, system_model_name)
                if system_model_configs[system_model_name].active:
                    subject.set_active(system_model_name)
                    active_model_name = system_model_name
                    break

    for classification, subject in classification_subjects.items():
        for model_name, observer in subject.observers.items():
            is_active = observer.active
            await SettingRepo.update_one(
                setting_schema.id,
                _set={'user_model_configs.$[elem].active': is_active},
                array_filters=[{'elem.name': model_name}],
            )

    return await get_active_model_config(setting_schema.id, system_model_configs, active_model_name)

async def get_current_models(
    request: Request,
    system_model_config: SystemModelConfigSchema = Depends(refresh_model_configs)
) -> List[LLM]:
    """Return the active model(s) of settings for current user"""
    models = [
        LLM_FACTORIES[endpoint['type']](**{
            'name': system_model_config.name,
            'description': system_model_config.description,
            'preprompt': system_model_config.preprompt,
            'classification': system_model_config.classification,
            'stream': system_model_config.stream,
            'parameters': dict(system_model_config.parameters),
            'server_kwargs': { 'headers': {'Authorization': f'Bearer {request.state.authorization}' }},
            'endpoint': endpoint,
        })
        for endpoint in system_model_config.endpoints
    ]
    return models

async def get_current_guardrails(
    request: Request, 
    system_model_configs: Dict[str, SystemModelConfigSchema] = Depends(load_system_model_config)
) -> List[LLM]:
    """Return optional LLM (if no guardrails LLM found, then guardrails is disabled)"""
    for _, system_model_config in system_model_configs.items():
        classification = system_model_config.classification
        strategy = get_strategy_for_classification(classification)
        
        if not strategy.is_content_safety():
            continue
        
        if system_model_config.active:
            return await get_current_models(request, system_model_config)

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
            'max_batch_tokens': active_model_config.max_batch_tokens,
            'max_client_batch_size': active_model_config.max_client_batch_size,
            'max_batch_requests': active_model_config.max_batch_requests,
            'num_workers': active_model_config.num_workers,
            'auto_truncate': active_model_config.auto_truncate,
            'token': request.state.authorization,
        })
        for endpoint in active_model_config.endpoints
    ]
    return models    

async def get_prompt_template(
    setting_schema: SettingSchema = Depends(get_user_settings),
    system_config_schema: SystemModelConfigSchema = Depends(refresh_model_configs)
) -> str:
    """Derive system prompt from either custom prompts or default system prompt"""
    if(
        active_user_config_id := next((config.id for config in setting_schema.user_model_configs if config.active), None)
    ) is not None:
        if(
            prompt := next((prompt for prompt in setting_schema.prompts for id in prompt.user_model_configs if id == active_user_config_id), None)
        ) is not None:
            return prompt.content
    
    return system_config_schema.preprompt or _DEFAULT_PREPROMPT